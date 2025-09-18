# FILE: simulation/contingency_planner.py
import logging
import uuid
import numpy as np
from typing import Dict, Any

from config import HUBS, RTH_BATTERY_THRESHOLD_FACTOR, DRONE_BASE_POWER_WATTS, DRONE_ADDITIONAL_WATTS_PER_KG, DELIVERY_MANEUVER_TIME_SEC
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
    
    # IMPLANTED LOGGING
    logging.critical(f"\n[CONTINGENCY] >>> TRIGGERING EMERGENCY for {drone_id} at t={state['simulation_time']:.1f}. Reason: {reason}. Current Status: {drone['status']} <<<\n")
    original_mission_id = drone['mission_id']
    
    original_mission = state['active_missions'].get(original_mission_id)
    log_event(state, f"⚠️ CONTINGENCY: {drone_id} entering EMERGENCY_RETURN due to: {reason}.")
    if original_mission:
        orders_returned = 0
        # Correctly iterate through all orders in the mission, not just stops
        for order_id in original_mission.get('order_ids', []):
            # Find the full order details from the stops list to requeue it
            order_details = next((s for s in original_mission.get('stops', []) if s['id'] == order_id), None)
            if order_details and order_id not in state['pending_orders']:
                state['pending_orders'][order_id] = order_details
                orders_returned += 1
        
        if orders_returned > 0:
            # IMPLANTED LOGGING
            logging.info(f"[CONTINGENCY] Returned {orders_returned} orders (IDs: {original_mission.get('order_ids', [])}) from failed mission {original_mission_id} to pending queue.")
        log_entry = { "mission_id": original_mission_id, "drone_id": drone_id, "completion_timestamp": state['simulation_time'], "outcome": f"Failed: {reason}", "planned_duration_sec": original_mission.get('total_planned_time', 0), "actual_duration_sec": state['simulation_time'] - original_mission.get('start_time', 0), "planned_energy_wh": original_mission.get('total_planned_energy', 0), "actual_energy_wh": original_mission.get('start_battery', 0) - drone['battery'], "number_of_stops": len(original_mission.get('stops', [])), }
        state['completed_missions_log'].append(log_entry)
        if original_mission_id in state['active_missions']:
            # IMPLANTED LOGGING
            logging.info(f"[CONTINGENCY] Deleting original mission '{original_mission_id}' from active_missions.")
            del state['active_missions'][original_mission_id]

    coord_manager = planners['coord_manager']
    hub_id, hub_pos = _find_nearest_hub(drone['pos'], coord_manager)
    if not hub_pos:
        log_event(state, f"CRITICAL: {drone_id} could not find a hub to return to!")
        drone['status'] = 'CRITICAL_FAILURE' 
        return
    
    # IMPLANTED LOGGING
    logging.info(f"[CONTINGENCY] Planning new emergency path for {drone_id} from {drone['pos']} to nearest hub {hub_id} at {hub_pos}.")
        
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
    # IMPLANTED LOGGING
    logging.info(f"[CONTINGENCY] Created new emergency mission '{emergency_mission_id}' for {drone_id}. Setting status to EMERGENCY_RETURN.")
    state['active_missions'][emergency_mission_id] = emergency_mission
    drone['status'] = 'EMERGENCY_RETURN'
    drone['mission_id'] = emergency_mission_id

def check_for_contingencies(state: Dict, planners: Dict, drone: Dict) -> bool:
    """
    Checks a single drone for low-battery or path invalidation contingencies.
    Returns True if a contingency was triggered, False otherwise.
    """
    # IMPLANTED LOGGING
    logging.info(f"[CONTINGENCY] Checking drone {drone['id']}. Status: {drone['status']}, Battery: {drone['battery']:.2f}Wh.")
    if drone['status'] not in ['EN ROUTE', 'RETURNING_TO_HUB', 'PERFORMING_DELIVERY']:
        return False

    mission = state['active_missions'].get(drone['mission_id'])
    if not mission: return False

    env = planners['env']
    predictor = planners['predictor']
    coord_manager = planners['coord_manager']
    drone_id = drone['id']

    # --- START: SOLUTION FOR FINAL BUG ---
    # Comprehensive energy calculation for the rest of the mission.
    energy_to_complete_mission = 0
    
    current_stop_idx = mission.get('current_stop_index', 0)
    stops = mission.get('stops', [])
    last_known_pos = drone['pos']
    payload_kg = mission.get('payload_kg', 0.0) # Assume payload for first leg

    if drone['status'] == 'PERFORMING_DELIVERY':
        remaining_maneuver_time_s = max(0, mission.get('maneuver_complete_at', 0) - state['simulation_time'])
        hover_power_watts = DRONE_BASE_POWER_WATTS + (payload_kg * DRONE_ADDITIONAL_WATTS_PER_KG)
        energy_to_complete_mission += (hover_power_watts * remaining_maneuver_time_s) / 3600.0
        last_known_pos = stops[current_stop_idx]['pos']
        # After this delivery, the payload is gone, and we move to the next stop index.
        payload_kg = 0 
        current_stop_idx += 1
    
    # Calculate energy for all remaining flight legs and maneuvers
    for i in range(current_stop_idx, len(stops)):
        next_stop_pos = stops[i]['pos']
        wind = env.weather.get_wind_at_location(*last_known_pos)
        _, leg_energy = predictor.predict(last_known_pos, next_stop_pos, payload_kg, wind)
        energy_to_complete_mission += leg_energy

        # Add energy for the delivery maneuver at this stop
        hover_power_watts = DRONE_BASE_POWER_WATTS + (payload_kg * DRONE_ADDITIONAL_WATTS_PER_KG)
        energy_to_complete_mission += (hover_power_watts * DELIVERY_MANEUVER_TIME_SEC) / 3600.0
        
        last_known_pos = next_stop_pos
        payload_kg = 0 # Payload is dropped after the first stop

    # Finally, calculate energy from the last stop to the final hub destination
    final_hub_pos = mission['destinations'][-1]
    wind = env.weather.get_wind_at_location(*last_known_pos)
    _, energy_to_hub = predictor.predict(last_known_pos, final_hub_pos, 0, wind)
    energy_to_complete_mission += energy_to_hub

    # We use the nearest hub for the absolute safety check (what if the final hub is too far?)
    _, nearest_hub_pos = _find_nearest_hub(drone['pos'], coord_manager)
    if nearest_hub_pos:
        return_wind = env.weather.get_wind_at_location(*drone['pos'])
        _, energy_to_return_to_hub = predictor.predict(drone['pos'], nearest_hub_pos, 0, return_wind)
        
        # The required energy is the larger of (A) finishing the mission, or (B) immediately returning to nearest hub
        required_energy_for_mission = energy_to_complete_mission * RTH_BATTERY_THRESHOLD_FACTOR
        required_energy_for_rth = energy_to_return_to_hub * RTH_BATTERY_THRESHOLD_FACTOR
        
        required_energy = max(required_energy_for_mission, required_energy_for_rth)
        trigger_emergency = drone['battery'] < required_energy
        
        # IMPLANTED LOGGING
        logging.info(
            f"[CONTINGENCY] Low Batt Check for {drone_id}: "
            f"Has {drone['battery']:.2f}Wh | "
            f"Needs to complete mission: {energy_to_complete_mission:.2f}Wh | "
            f"Needs for immediate RTH: {energy_to_return_to_hub:.2f}Wh | "
            f"Required with safety factor ({RTH_BATTERY_THRESHOLD_FACTOR}x): {required_energy:.2f}Wh. "
            f"Trigger Emergency: {trigger_emergency}"
        )
        # --- END: SOLUTION FOR FINAL BUG ---
        
        if trigger_emergency:
            _trigger_emergency_return(state, drone_id, "Critically Low Battery", planners)
            return True

    # --- Path Invalidation Check ---
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
                    return True
    
    return False