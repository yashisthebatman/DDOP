# FILE: simulation/deconfliction.py
import numpy as np
from utils.geometry import calculate_distance_3d
from config import SAFETY_BUBBLE_RADIUS_METERS, AVOIDANCE_MANEUVER_ALTITUDE_SEP, MAX_ALTITUDE, MIN_ALTITUDE

def check_and_resolve_conflicts(active_drones: dict, planners: dict):
    """Checks all pairs of drones for proximity and initiates intelligent avoidance."""
    drone_ids = list(active_drones.keys())
    for i in range(len(drone_ids)):
        for j in range(i + 1, len(drone_ids)):
            d1_id, d2_id = drone_ids[i], drone_ids[j]
            d1, d2 = active_drones[d1_id], active_drones[d2_id]

            # Only deconflict drones that are currently in flight and NOT already avoiding.
            if d1['status'] not in ['EN ROUTE', 'EMERGENCY_RETURN', 'RETURNING_TO_HUB'] or \
               d2['status'] not in ['EN ROUTE', 'EMERGENCY_RETURN', 'RETURNING_TO_HUB']:
                continue

            d1_pos_m = planners['coord_manager'].world_to_meters(d1['pos'])
            d2_pos_m = planners['coord_manager'].world_to_meters(d2['pos'])
            dist = calculate_distance_3d(d1_pos_m, d2_pos_m)

            if dist < SAFETY_BUBBLE_RADIUS_METERS:
                # CONFLICT DETECTED!
                # Drone with the alphabetically smaller ID prefers to climb.
                if d1_id < d2_id:
                    initiate_avoidance(d1, "climb", planners)
                    initiate_avoidance(d2, "descend", planners)
                else:
                    initiate_avoidance(d1, "descend", planners)
                    initiate_avoidance(d2, "climb", planners)

def initiate_avoidance(drone: dict, preferred_maneuver: str, planners: dict):
    """
    Sets a drone to AVOIDING with a new temporary target, ensuring the target
    is not inside a known obstacle.
    """
    if drone['status'] == 'AVOIDING': return

    env = planners['env']
    coord_manager = planners['coord_manager']
    
    drone['original_status_before_avoid'] = drone.get('status', 'EN ROUTE')
    drone['status'] = 'AVOIDING'
    
    current_pos_m = coord_manager.world_to_meters(drone['pos'])
    
    # Prioritized list of maneuvers to try
    maneuvers = ["climb", "descend", "lateral_north", "lateral_east"]
    if preferred_maneuver in maneuvers:
        maneuvers.insert(0, maneuvers.pop(maneuvers.index(preferred_maneuver)))

    for maneuver in maneuvers:
        target_pos_m = list(current_pos_m)
        if maneuver == "climb":
            target_pos_m[2] = min(MAX_ALTITUDE, current_pos_m[2] + AVOIDANCE_MANEUVER_ALTITUDE_SEP)
        elif maneuver == "descend":
            target_pos_m[2] = max(MIN_ALTITUDE, current_pos_m[2] - AVOIDANCE_MANEUVER_ALTITUDE_SEP)
        elif maneuver == "lateral_north": # Positive Y in meter-space
            target_pos_m[1] += SAFETY_BUBBLE_RADIUS_METERS
        elif maneuver == "lateral_east": # Positive X in meter-space
            target_pos_m[0] += SAFETY_BUBBLE_RADIUS_METERS
        
        target_pos_world = coord_manager.meters_to_world(tuple(target_pos_m))
        
        # Check if the proposed avoidance point is safe
        if not env.is_point_obstructed(target_pos_world):
            drone['avoidance_target_pos'] = (float(target_pos_world[0]), float(target_pos_world[1]), float(target_pos_world[2]))
            return # Safe maneuver found and set

    # If no safe maneuver is found, the drone will not enter AVOIDING state.
    # It will continue its path and rely on contingency planner to detect if it gets stuck.
    drone['status'] = drone['original_status_before_avoid'] # Revert status
    logging.warning(f"CRITICAL DECONFLICTION: Drone {drone['id']} could not find a safe avoidance maneuver.")