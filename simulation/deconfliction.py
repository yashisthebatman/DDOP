# FILE: simulation/deconfliction.py
from utils.geometry import calculate_distance_3d
from config import SAFETY_BUBBLE_RADIUS_METERS, AVOIDANCE_MANEUVER_ALTITUDE_SEP, MAX_ALTITUDE, MIN_ALTITUDE

def check_and_resolve_conflicts(active_drones: dict, coord_manager):
    """Checks all pairs of drones for proximity and initiates avoidance."""
    drone_ids = list(active_drones.keys())
    for i in range(len(drone_ids)):
        for j in range(i + 1, len(drone_ids)):
            d1_id, d2_id = drone_ids[i], drone_ids[j]
            d1, d2 = active_drones[d1_id], active_drones[d2_id]

            # Only deconflict drones that are currently in flight and NOT already avoiding.
            if d1['status'] not in ['EN ROUTE', 'EMERGENCY_RETURN'] or d2['status'] not in ['EN ROUTE', 'EMERGENCY_RETURN']:
                continue

            d1_pos_m = coord_manager.world_to_meters(d1['pos'])
            d2_pos_m = coord_manager.world_to_meters(d2['pos'])
            dist = calculate_distance_3d(d1_pos_m, d2_pos_m)

            if dist < SAFETY_BUBBLE_RADIUS_METERS:
                # CONFLICT DETECTED!
                # Simple rule: drone with the alphabetically smaller ID climbs.
                if d1_id < d2_id:
                    initiate_avoidance(d1, "climb")
                    initiate_avoidance(d2, "descend")
                else:
                    initiate_avoidance(d1, "descend")
                    initiate_avoidance(d2, "climb")

def initiate_avoidance(drone: dict, maneuver: str):
    """Sets a drone's state to AVOIDING with a new temporary target."""
    # This check prevents a drone already avoiding from being re-triggered.
    if drone['status'] == 'AVOIDING':
        return

    drone['original_status_before_avoid'] = drone.get('status', 'EN ROUTE')
    drone['status'] = 'AVOIDING'
    
    current_pos = list(drone['pos'])
    altitude_change = AVOIDANCE_MANEUVER_ALTITUDE_SEP

    if maneuver == "climb":
        new_alt = min(MAX_ALTITUDE, current_pos[2] + altitude_change)
        # FIX: Ensure position is a tuple of standard Python floats to prevent JSON errors.
        drone['avoidance_target_pos'] = (float(current_pos[0]), float(current_pos[1]), float(new_alt))
    else: # descend
        new_alt = max(MIN_ALTITUDE, current_pos[2] - altitude_change)
        # FIX: Ensure position is a tuple of standard Python floats.
        drone['avoidance_target_pos'] = (float(current_pos[0]), float(current_pos[1]), float(new_alt))