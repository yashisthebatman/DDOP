# FILE: fleet/manager.py
import logging
from typing import Dict, List, Tuple
from fleet.cbs_components import Agent
from planners.cbsh_planner import CBSHPlanner
from ml_predictor.predictor import EnergyTimePredictor
from config import RTH_BATTERY_THRESHOLD_FACTOR, DELIVERY_MANEUVER_TIME_SEC
from utils.geometry import calculate_distance_3d

class Mission:
    def __init__(self, mission_id, drone_id, start_pos, destinations, payload_kg, order_ids, optimization_weights={}):
        self.mission_id = mission_id
        self.drone_id = drone_id
        self.start_pos = start_pos
        self.destinations = destinations
        self.payload_kg = payload_kg
        self.order_ids = order_ids
        self.optimization_weights = optimization_weights
        self.stops = [] # Store full order details
        self.start_hub = None
        self.end_hub = None
        self.is_paused = False
        
        self.path, self.path_world_coords = [], []
        self.total_planned_energy, self.total_planned_time = 0.0, 0.0
        self.state = "PENDING"
        self.current_stop_index = 0
        self.mission_time_elapsed = 0.0
        self.flight_time_elapsed = 0.0

    def to_dict(self):
        """Converts mission object to a dictionary for JSON serialization."""
        return {
            'mission_id': self.mission_id,
            'drone_id': self.drone_id,
            'order_ids': self.order_ids,
            'stops': self.stops,
            'start_hub': self.start_hub,
            'end_hub': self.end_hub,
            'is_paused': self.is_paused,
            'start_pos': self.start_pos,
            'destinations': self.destinations,
            'payload_kg': self.payload_kg,
            'path_world_coords': self.path_world_coords,
            'total_planned_energy': self.total_planned_energy,
            'total_planned_time': self.total_planned_time,
            'start_time': 0, 
            'start_battery': 0,
            'current_stop_index': self.current_stop_index,
            'mission_time_elapsed': self.mission_time_elapsed,
            'flight_time_elapsed': self.flight_time_elapsed,
            'total_maneuver_time': 0
        }

class FleetManager:
    def __init__(self, cbs_planner: CBSHPlanner, predictor: EnergyTimePredictor):
        self.cbs_planner = cbs_planner
        self.predictor = predictor

    def plan_pending_missions(self, state: Dict) -> Tuple[bool, Dict]:
        """
        Finds all drones in 'PLANNING' state, runs the CBSH planner, and
        returns a dictionary of updates to be applied to the main state, rather
        than modifying the state directly. This makes it thread-safe.
        """
        missions_to_plan = []
        drones_in_planning = []
        for drone_id, drone in state['drones'].items():
            if drone['status'] == 'PLANNING':
                mission_id = drone.get('mission_id')
                if mission_id and mission_id in state['active_missions']:
                    missions_to_plan.append(state['active_missions'][mission_id])
                    drones_in_planning.append(drone_id)

        if not missions_to_plan:
            return True, {"message": "No missions in PLANNING state."}

        active_agents: List[Agent] = [
            Agent(id=m['drone_id'], start_pos=m['start_pos'], goal_pos=m['destinations'][-1], config={'payload_kg': m['payload_kg']})
            for m in missions_to_plan
        ]
        
        if not active_agents:
            return True, {"message": "No active agents to plan for."}

        logging.info(f"FleetManager initiating CBSH planning for {len(active_agents)} agents.")
        solution = self.cbs_planner.plan_fleet(active_agents)

        drone_updates = {}
        mission_updates = {}
        successful_mission_ids = []
        mission_failures = []
        
        if not solution:
            logging.error("CBSH planning FAILED for the fleet.")
            for agent_id in drones_in_planning:
                drone_updates[agent_id] = {'status': 'IDLE', 'mission_id': None}
                mission_id = state['drones'][agent_id].get('mission_id')
                if mission_id: mission_failures.append(mission_id)
            return False, {"drone_updates": drone_updates, "mission_failures": mission_failures, "error": "CBS could not find a solution."}

        logging.info("CBSH planning successful. Preparing mission updates.")
        
        solved_agents = set(solution.keys())
        for agent_id, path in solution.items():
            drone = state['drones'][agent_id]
            mission_id = drone['mission_id']
            mission = state['active_missions'][mission_id]

            world_path = [p[0] for p in path]
            smoothed_path = self.cbs_planner.smoother.smooth_path(world_path, self.cbs_planner.env)
            
            total_energy = 0
            if path and len(path) > 1:
                for i in range(len(world_path) - 1):
                    p1, p2 = world_path[i], world_path[i+1]
                    wind = self.cbs_planner.env.weather.get_wind_at_location(*p1)
                    _, energy_pred = self.predictor.predict(p1, p2, mission['payload_kg'], wind, world_path[i-1] if i>0 else None)
                    total_energy += energy_pred
                
                num_stops = len(mission.get('stops', []))
                total_maneuver_time = num_stops * DELIVERY_MANEUVER_TIME_SEC
                flight_time = path[-1][1]
                
                mission_updates[mission_id] = {
                    'path_world_coords': smoothed_path,
                    'total_planned_energy': total_energy,
                    'total_planned_time': flight_time + total_maneuver_time,
                    'total_maneuver_time': total_maneuver_time,
                    'start_time': state['simulation_time'],
                    'start_battery': drone['battery']
                }
                drone_updates[agent_id] = {'status': 'EN ROUTE'}
                successful_mission_ids.append(mission_id)
            else:
                drone_updates[agent_id] = {'status': 'IDLE', 'mission_id': None}
                mission_failures.append(mission_id)

        unsolved_agents = [agent_id for agent_id in drones_in_planning if agent_id not in solved_agents]
        for agent_id in unsolved_agents:
            drone_updates[agent_id] = {'status': 'IDLE', 'mission_id': None}
            mission_id = state['drones'][agent_id].get('mission_id')
            if mission_id: mission_failures.append(mission_id)
            
        return True, {"drone_updates": drone_updates, "mission_updates": mission_updates, "successful_mission_ids": successful_mission_ids, "mission_failures": mission_failures}