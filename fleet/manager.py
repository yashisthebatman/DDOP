# FILE: fleet/manager.py
import logging
from typing import Dict, List, Tuple
from fleet.cbs_components import Agent
from planners.cbsh_planner import CBSHPlanner
from ml_predictor.predictor import EnergyTimePredictor
from config import RTH_BATTERY_THRESHOLD_FACTOR
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
        
        self.path, self.path_world_coords = [], []
        self.total_planned_energy, self.total_planned_time = 0.0, 0.0
        self.state = "PENDING"
        self.current_leg_index = 0

    def to_dict(self):
        """Converts mission object to a dictionary for JSON serialization."""
        return {
            'mission_id': self.mission_id,
            'drone_id': self.drone_id,
            'order_ids': self.order_ids,
            'start_pos': self.start_pos,
            'destinations': self.destinations,
            'payload_kg': self.payload_kg,
            'path_world_coords': self.path_world_coords,
            'total_planned_energy': self.total_planned_energy,
            'total_planned_time': self.total_planned_time,
            'start_time': 0, 
            'start_battery': 0
        }

    def is_complete(self):
        return self.current_leg_index >= len(self.destinations)

    def advance_leg(self):
        if not self.is_complete():
            self.current_leg_index += 1
            self.state = "COMPLETED" if self.is_complete() else "PENDING"

class FleetManager:
    def __init__(self, cbs_planner: CBSHPlanner, predictor: EnergyTimePredictor):
        self.cbs_planner = cbs_planner
        self.predictor = predictor

    def plan_pending_missions(self, state: Dict) -> Tuple[bool, Dict]:
        """
        Finds all drones in a 'PLANNING' state and attempts to plan a deconflicted
        fleet-wide solution for them.
        """
        missions_to_plan = []
        drones_in_planning = []
        for drone_id, drone in state['drones'].items():
            if drone['status'] == 'PLANNING':
                mission_id = drone.get('mission_id')
                if mission_id and mission_id in state['active_missions']:
                    missions_to_plan.append(state['active_missions'][mission_id])
                    drones_in_planning.append(drone_id)
                else:
                    logging.warning(f"Drone {drone_id} is PLANNING but has no valid mission. Reverting to IDLE.")
                    drone['status'] = 'IDLE'

        if not missions_to_plan:
            return True, {"message": "No missions in PLANNING state."}

        active_agents: List[Agent] = []
        for m in missions_to_plan:
            config = {'payload_kg': m['payload_kg']}
            agent = Agent(id=m['drone_id'], start_pos=m['start_pos'], goal_pos=m['destinations'][-1], config=config)
            active_agents.append(agent)

        if not active_agents:
            return True, {"message": "No active agents to plan for."}
        
        logging.info(f"FleetManager initiating CBSH planning for {len(active_agents)} agents.")
        solution = self.cbs_planner.plan_fleet(active_agents)
        
        if not solution:
            logging.error("CBSH planning FAILED for the fleet. Reverting drones to IDLE.")
            for agent_id in drones_in_planning:
                drone = state['drones'][agent_id]
                drone['status'] = 'IDLE'
                mission_id = drone.get('mission_id')
                if mission_id and mission_id in state['active_missions']:
                    mission = state['active_missions'][mission_id]
                    logging.error(f"Orders from failed mission {mission_id} need to be re-queued: {mission['order_ids']}")
                    # A robust implementation would re-create these in pending_orders, but this is complex
                    # as the original order objects are gone. For now, we delete the failed mission.
                    del state['active_missions'][mission_id]
                drone['mission_id'] = None
            return False, {"error": "CBS could not find a solution for the fleet."}

        logging.info("CBSH planning successful. Updating missions.")
        
        for agent_id, path in solution.items():
            drone = state['drones'][agent_id]
            mission_id = drone['mission_id']
            mission = state['active_missions'][mission_id]

            world_path = [p[0] for p in path]
            smoothed_path = self.cbs_planner.smoother.smooth_path(world_path, self.cbs_planner.env)
            
            if path and len(path) > 1:
                total_energy, p_prev = 0, None
                for i in range(len(world_path) - 1):
                    p1, p2 = world_path[i], world_path[i+1]
                    wind = self.cbs_planner.env.weather.get_wind_at_location(*p1)
                    _, energy_pred = self.predictor.predict(p1, p2, mission['payload_kg'], wind, p_prev)
                    total_energy += energy_pred
                    p_prev = p1
                
                mission['path_world_coords'] = smoothed_path
                mission['total_planned_energy'] = total_energy
                mission['total_planned_time'] = path[-1][1]
                mission['start_time'] = state['simulation_time']
                mission['start_battery'] = drone['battery']
                
                drone['status'] = 'EN ROUTE'
                logging.info(f"Drone {agent_id} is now EN ROUTE for mission {mission_id}.")
            else:
                logging.warning(f"Planning for {agent_id} resulted in an invalid path. Reverting to IDLE.")
                drone['status'] = 'IDLE'
        
        return True, {"message": "Planning cycle completed."}