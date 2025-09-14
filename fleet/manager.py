# FILE: fleet/manager.py
import logging
import queue
import threading
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
            'total_maneuver_time': 0,
            'current_path_target_index': 1 
        }

class FleetManager:
    def __init__(self, cbs_planner: CBSHPlanner, predictor: EnergyTimePredictor, state_lock: threading.Lock):
        self.cbs_planner = cbs_planner
        self.predictor = predictor
        self.state_lock = state_lock
        self.planning_queue = queue.PriorityQueue()

    def add_mission_to_queue(self, mission: Mission):
        is_high_priority = any(o.get('high_priority', False) for o in mission.stops)
        priority = 0 if is_high_priority else 1
        self.planning_queue.put((priority, mission))
        logging.info(f"Enqueued mission {mission.mission_id} with priority {priority}.")

    def plan_pending_missions(self, state: Dict) -> Tuple[bool, Dict]:
        try:
            _, mission = self.planning_queue.get_nowait()
        except queue.Empty:
            return True, {}

        mission_id = mission.mission_id
        drone_id = mission.drone_id

        with self.state_lock:
            if state['drones'][drone_id]['status'] != 'IDLE':
                logging.warning(f"Drone {drone_id} is no longer IDLE. Aborting planning for {mission_id}.")
                return False, {"mission_failures": [mission_id]}
            
            state['active_missions'][mission_id] = mission.to_dict()
            state['drones'][drone_id]['status'] = 'PLANNING'
            state['drones'][drone_id]['mission_id'] = mission_id
        
        active_agent = Agent(id=drone_id, start_pos=mission.start_pos, goal_pos=mission.destinations[-1], config={'payload_kg': mission.payload_kg})
        
        logging.info(f"FleetManager initiating CBSH planning for agent {drone_id} on mission {mission_id}.")
        solution = self.cbs_planner.plan_fleet([active_agent])

        drone_updates = {}
        mission_updates = {}
        
        if not solution or drone_id not in solution or not solution[drone_id]:
            logging.error(f"CBSH planning FAILED for agent {drone_id} on mission {mission_id}.")
            # Return the full drone object to prevent data corruption
            drone_updates[drone_id] = state['drones'][drone_id].copy()
            drone_updates[drone_id]['status'] = 'IDLE'
            drone_updates[drone_id]['mission_id'] = None
            return False, {"drone_updates": drone_updates, "mission_failures": [mission_id], "error": "CBS could not find a solution."}

        logging.info(f"CBSH planning successful for {drone_id}. Preparing mission updates.")
        
        path = solution[drone_id]
        world_path = [p[0] for p in path]
        smoothed_path = self.cbs_planner.smoother.smooth_path(world_path, self.cbs_planner.env)
        
        total_energy = 0
        if path and len(path) > 1:
            mission_dict = state['active_missions'][mission_id]
            for i in range(len(world_path) - 1):
                p1, p2 = world_path[i], world_path[i+1]
                wind = self.cbs_planner.env.weather.get_wind_at_location(*p1)
                _, energy_pred = self.predictor.predict(p1, p2, mission_dict['payload_kg'], wind, world_path[i-1] if i>0 else None)
                total_energy += energy_pred
            
            num_stops = len(mission_dict.get('stops', []))
            total_maneuver_time = num_stops * DELIVERY_MANEUVER_TIME_SEC
            flight_time = path[-1][1]
            
            mission_updates[mission_id] = {
                'path_world_coords': smoothed_path,
                'total_planned_energy': total_energy,
                'total_planned_time': flight_time + total_maneuver_time,
                'total_maneuver_time': total_maneuver_time,
                'start_time': state['simulation_time'],
                'start_battery': state['drones'][drone_id]['battery'],
                'current_path_target_index': 1
            }
            # Return the full drone object with updates to prevent corruption
            drone_updates[drone_id] = state['drones'][drone_id].copy()
            drone_updates[drone_id]['status'] = 'EN ROUTE'
            return True, {"drone_updates": drone_updates, "mission_updates": mission_updates, "successful_mission_ids": [mission_id]}
        else:
            drone_updates[drone_id] = state['drones'][drone_id].copy()
            drone_updates[drone_id]['status'] = 'IDLE'
            drone_updates[drone_id]['mission_id'] = None
            return False, {"drone_updates": drone_updates, "mission_failures": [mission_id]}