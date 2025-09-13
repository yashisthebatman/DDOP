# FILE: simulation/contingency_planner.py
import logging
import uuid
import numpy as np
from typing import Dict, Any

from config import HUBS, RTH_BATTERY_THRESHOLD_FACTOR
from utils.geometry import calculate_distance_3d
from planners.single_agent_planner import SingleAgentPlanner

def log_event(state, message):
    """Adds a new message to the persistent event log."""
    import time
    state['log'].insert(0, f"{time.strftime('%H:%M:%S')} - {message}")

def _find_nearest_hub(pos, coord_manager):
    """Finds the closest hub to a given world position."""
    pos_m = coord_manager.world_to_meters(pos)
    hubs_with_dist = []
    for hub_name, hub_pos in HUBS.items():
        hub_pos_m = coord_manager.world_to_meters(hub_pos)
        dist = calculate_distance_3d(pos_m, hub_pos_m)
        hubs_with_dist.append((dist, hub_name, hub_pos))
    
    if not hubs_with_dist: return None, None
    
    _, nearest_hub_name, nearest_hub_pos = min(hubs_with_dist, key=lambda x: x[0])
    return nearest_hub_name, nearest_hub_pos

def _trigger_emergency_return(state: Dict, drone_id: str, reason: str, planners: Dict):
    """Handles the full logic for cancelling a mission and planning an RTH."""
    drone = state['drones'][drone_id]
    original_mission_id = drone['mission_id']
    original_mission = state['active_missions'].get(original_mission_id)

    log_event(state, f"⚠️ CONTINGENCY: {drone_id} entering EMERGENCY_RETURN due to: {reason}.")

    # 1. Cancel original mission and return its orders to the pending queue
    if original_mission:
        orders_returned = 0
        for order_details in original_mission.get('stops', []):
            if order_details['id'] not in state['pending_orders']:
                state['pending_orders'][order_details['id']] = order_details
                orders_returned += 1
        if orders_returned > 0:
            log_event(state, f"Returned {orders_returned} orders from failed mission {original_mission_id} to queue.")
        
        # Log the failure for analytics
        log_entry = {
            "mission_id": original_mission_id, "drone_id": drone_id,
            "completion_timestamp": state['simulation_time'], "outcome": f"Failed: {reason}",
            "planned_duration_sec": original_mission.get('total_planned_time', 0), 
            "actual_duration_sec": state['simulation_time'] - original_mission.get('start_time', 0),
            "planned_energy_wh": original_mission.get('total_planned_energy', 0),
            "actual_energy_wh": original_mission.get('start_battery', 0) - drone['battery'], 
            "number_of_stops": len(original_mission.get('stops', [])),
        }
        state['completed_missions_log'].append(log_entry)
        del state['active_missions'][original_mission_id]

    # 2. Plan a new, high-priority path back to the nearest hub
    coord_manager = planners['coord_manager']
    _, hub_pos = _find_nearest_hub(drone['pos'], coord_manager)
    if not hub_pos:
        log_event(state, f"CRITICAL: {drone_id} could not find a hub to return to!")
        drone['status'] = 'IDLE' # Drone is lost, requires manual recovery
        return

    planner = SingleAgentPlanner(planners['env'], planners['predictor'], coord_manager)
    path, status = planner.find_strategic_path_rrt(drone['pos'], hub_pos)

    if not path:
        log_event(state, f"CRITICAL: {drone_id} could not plan emergency return path! Status: {status}")
        drone['status'] = 'IDLE' # Effectively a crash/manual recovery
        return

    # 3. Create and assign the new emergency mission
    total_time, total_energy = 0, 0
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            t, e = planners['predictor'].predict(p1, p2, 0, [0,0,0])
            total_time += t
            total_energy += e

    emergency_mission_id = f"EM-{uuid.uuid4().hex[:6]}"
    emergency_mission = {
        'mission_id': emergency_mission_id, 'drone_id': drone_id, 'order_ids': [], 'stops': [],
        'start_pos': drone['pos'], 'destinations': [hub_pos], 'payload_kg': 0,
        'path_world_coords': path, 'total_planned_time': total_time, 'total_planned_energy': total_energy,
        'start_time': state['simulation_time'], 'start_battery': drone['battery'],
        'mission_time_elapsed': 0.0, 'flight_time_elapsed': 0.0, 'total_maneuver_time': 0,
        'end_hub': _find_nearest_hub(hub_pos, coord_manager)[0]
    }
    state['active_missions'][emergency_mission_id] = emergency_mission
    drone['status'] = 'EMERGENCY_RETURN'
    drone['mission_id'] = emergency_mission_id

def check_for_contingencies(state: Dict, planners: Dict):
    """
    Checks all active drones for conditions that require an emergency RTH.
    """
    env = planners['env']
    predictor = planners['predictor']
    coord_manager = planners['coord_manager']
    
    drones_to_check = [d for d in state['drones'].values() if d['status'] == 'EN ROUTE']

    for drone in drones_to_check:
        drone_id = drone['id']
        mission = state['active_missions'].get(drone['mission_id'])
        if not mission: continue
        
        # --- Check 1: Low Battery ---
        # Predict energy to finish current mission AND return to nearest hub afterwards
        final_dest = mission['destinations'][-1]
        _, nearest_hub_pos_after_mission = _find_nearest_hub(final_dest, coord_manager)
        
        if nearest_hub_pos_after_mission:
            _, energy_to_finish_mission = predictor.predict(drone['pos'], final_dest, mission['payload_kg'], [0,0,0])
            _, energy_to_return_to_hub = predictor.predict(final_dest, nearest_hub_pos_after_mission, 0, [0,0,0])
            
            # FIX: Use the factor from config file instead of a hardcoded value.
            required_energy = (energy_to_finish_mission + energy_to_return_to_hub) * RTH_BATTERY_THRESHOLD_FACTOR
            
            if drone['battery'] < required_energy:
                _trigger_emergency_return(state, drone_id, "Critically Low Battery", planners)
                continue # Drone is now in emergency, skip other checks for it

        # --- Check 2: Path Invalidated by New NFZ ---
        if env.was_nfz_just_added:
            path = mission.get('path_world_coords', [])
            if path:
                current_pos_np = np.array(drone['pos'])
                distances = [np.linalg.norm(current_pos_np - np.array(p)) for p in path]
                current_idx = np.argmin(distances)

                # Check the remainder of the path for obstructions
                for i in range(current_idx, len(path) - 1):
                    if env.is_line_obstructed(path[i], path[i+1]):
                        _trigger_emergency_return(state, drone_id, "Path Invalidated by NFZ", planners)
                        break # Path is bad, move to next drone

    # Reset the flag after all drones have been checked for this tick
    if env.was_nfz_just_added:
        env.was_nfz_just_added = False