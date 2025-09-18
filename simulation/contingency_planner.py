# FILE: simulation/contingency_planner.py
import logging
import uuid
import numpy as np
from typing import Dict, Any

from config import HUBS, RTH_BATTERY_THRESHOLD_FACTOR
from utils.geometry import calculate_distance_3d
from planners.single_agent_planner import SingleAgentPlanner

def log_event(state, message):
    import time
    state['log'].insert(0, f"{time.strftime('%H:%M:%S')} - {message}")

def _find_nearest_hub(pos, coord_manager):
    """Finds the closest hub to a given world position."""
    pos_m = coord_manager.world_to_meters(pos)
    hubs_with_dist = []
    for hub in HUBS:
        hub_pos_m = coord_manager.world_to_meters(hub['location'])
        dist = calculate_distance_3d(pos_m, hub_pos_m)
        hubs_with_dist.append((dist, hub['id'], hub['location']))
    
    if not hubs_with_dist: return None, None
    
    _, nearest_hub_id, nearest_hub_pos = min(hubs_with_dist, key=lambda x: x[0])
    return nearest_hub_id, nearest_hub_pos

def _trigger_emergency_return(state: Dict, drone_id: str, reason: str, planners: Dict):
    drone = state['drones'][drone_id]
    if drone['status'] in ['EMERGENCY_RETURN', 'CRITICAL_FAILURE']:
        return
        
    original_mission_id = drone['mission_id']
    logging.info(f"[DEBUG] Tick {state['simulation_time']:.1f}: CONTINGENCY TRIGGERED for {drone_id} on mission {original_mission_id}. Reason: {reason}. Current status: {drone['status']}")
    
    original_mission = state['active_missions'].get(original_mission_id)
    log_event(state, f"⚠️ CONTINGENCY: {drone_id} entering EMERGENCY_RETURN due to: {reason}.")
    if original_mission:
        orders_returned = 0
        for order_details in original_mission.get('stops', []):
            if order_details['id'] not in state['pending_orders']:
                state['pending_orders'][order_details['id']] = order_details
                orders_returned += 1
        if orders_returned > 0:
            log_event(state, f"Returned {orders_returned} orders from failed mission {original_mission_id} to queue.")
        log_entry = { "mission_id": original_mission_id, "drone_id": drone_id, "completion_timestamp": state['simulation_time'], "outcome": f"Failed: {reason}", "planned_duration_sec": original_mission.get('total_planned_time', 0), "actual_duration_sec": state['simulation_time'] - original_mission.get('start_time', 0), "planned_energy_wh": original_mission.get('total_planned_energy', 0), "actual_energy_wh": original_mission.get('start_battery', 0) - drone['battery'], "number_of_stops": len(original_mission.get('stops', [])), }
        state['completed_missions_log'].append(log_entry)
        if original_mission_id in state['active_missions']:
            del state['active_missions'][original_mission_id]

    coord_manager = planners['coord_manager']
    hub_id, hub_pos = _find_nearest_hub(drone['pos'], coord_manager)
    if not hub_pos:
        log_event(state, f"CRITICAL: {drone_id} could not find a hub to return to!")
        drone['status'] = 'CRITICAL_FAILURE' 
        return
        
    planner = SingleAgentPlanner(planners['env'], planners['predictor'], coord_manager)
    path, status = planner.find_strategic_path_rrt(drone['pos'], hub_pos)
    if not path:
        log_event(state, f"CRITICAL: {drone_id} could not plan emergency return path! Status: {status}")
        drone['status'] = 'CRITICAL_FAILURE'
        return
        
    total_time, total_energy = 0, 0
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            wind = planners['env'].weather.get_wind_at_location(*p1)
            t, e = planners['predictor'].predict(p1, p2, 0, wind)
            total_time += t
            total_energy += e
            
    emergency_mission_id = f"EM-{uuid.uuid4().hex[:6]}"
    emergency_mission = { 'mission_id': emergency_mission_id, 'drone_id': drone_id, 'order_ids': [], 'stops': [], 'start_pos': drone['pos'], 'destinations': [hub_pos], 'payload_kg': 0, 'path_world_coords': path, 'total_planned_time': total_time, 'total_planned_energy': total_energy, 'start_time': state['simulation_time'], 'start_battery': drone['battery'], 'mission_time_elapsed': 0.0, 'flight_time_elapsed': 0.0, 'total_maneuver_time': 0, 'end_hub': hub_id }
    state['active_missions'][emergency_mission_id] = emergency_mission
    drone['status'] = 'EMERGENCY_RETURN'
    drone['mission_id'] = emergency_mission_id

def check_for_contingencies(state: Dict, planners: Dict):
    # ... (rest of file is the same)
    env = planners['env']
    predictor = planners['predictor']
    coord_manager = planners['coord_manager']
    drones_to_check = [d for d in state['drones'].values() if d['status'] in ['EN ROUTE', 'RETURNING_TO_HUB']]
    for drone in drones_to_check:
        drone_id = drone['id']
        mission = state['active_missions'].get(drone['mission_id'])
        if not mission: continue
        final_dest = mission['destinations'][-1]
        _, nearest_hub_pos_after_mission = _find_nearest_hub(final_dest, coord_manager)
        if nearest_hub_pos_after_mission:
            current_wind = env.weather.get_wind_at_location(*drone['pos'])
            # MODIFIED: Use .get() to safely access payload_kg, defaulting to 0.0.
            payload = mission.get('payload_kg', 0.0)
            _, energy_to_finish_mission = predictor.predict(drone['pos'], final_dest, payload, current_wind)
            _, energy_to_return_to_hub = predictor.predict(final_dest, nearest_hub_pos_after_mission, 0, current_wind)
            required_energy = (energy_to_finish_mission + energy_to_return_to_hub) * RTH_BATTERY_THRESHOLD_FACTOR
            if drone['battery'] < required_energy:
                _trigger_emergency_return(state, drone_id, "Critically Low Battery", planners)
                continue
        if env.was_nfz_just_added:
            path = mission.get('path_world_coords', [])
            if path:
                current_pos_np = np.array(drone['pos'])
                distances = [np.linalg.norm(current_pos_np - np.array(p)) for p in path]
                current_idx = np.argmin(distances)
                start_check_idx = max(0, current_idx - 1)
                for i in range(start_check_idx, len(path) - 1):
                    if env.is_line_obstructed(path[i], path[i+1]):
                        _trigger_emergency_return(state, drone_id, "Path Invalidated by NFZ", planners)
                        break
    if env.was_nfz_just_added:
        env.was_nfz_just_added = False