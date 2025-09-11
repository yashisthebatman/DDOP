import numpy as np

def calculate_distance_3d(p1, p2):
    """Calculates the Euclidean distance between two 3D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_vector_angle_3d(v1, v2):
    """Calculates the angle in radians between two 3D vectors."""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    v1_u = v1 / norm_v1
    v2_u = v2 / norm_v2
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot_product)

def calculate_wind_effect(flight_vector, wind_vector, drone_speed):
    """Returns a tuple of (time_impact_factor, energy_impact_factor)."""
    norm_flight_vector = np.linalg.norm(flight_vector)
    if norm_flight_vector == 0:
        return 1.0, 1.0
    flight_unit_vector = flight_vector / norm_flight_vector
    wind_component = np.dot(wind_vector, flight_unit_vector)
    effective_speed = drone_speed + wind_component
    if effective_speed <= 1e-6:
        return float('inf'), float('inf')
    time_impact_factor = drone_speed / effective_speed
    energy_impact_factor = (drone_speed / effective_speed)**2
    return time_impact_factor, energy_impact_factor

def line_segment_intersects_aabb(p1, p2, box_bounds):
    """Performs efficient 3D line segment vs. AABB intersection test."""
    box_min = np.array([box_bounds[0], box_bounds[1], box_bounds[2]])
    box_max = np.array([box_bounds[3], box_bounds[4], box_bounds[5]])
    line_start, line_end = np.array(p1), np.array(p2)
    
    if (np.all(line_start >= box_min) and np.all(line_start <= box_max)) or \
       (np.all(line_end >= box_min) and np.all(line_end <= box_max)):
        return True
    
    direction = line_end - line_start
    if np.allclose(direction, 0):
        return False
    
    inv_direction = np.divide(1.0, direction, where=direction!=0, out=np.full_like(direction, np.inf, dtype=float))
    t_near = (box_min - line_start) * inv_direction
    t_far = (box_max - line_start) * inv_direction
    
    tmin = np.max(np.minimum(t_near, t_far))
    tmax = np.min(np.maximum(t_near, t_far))
    
    return tmax >= max(0, tmin) and tmin <= 1

def point_in_aabb(point, box_bounds):
    """Check if a point is inside an AABB."""
    x, y, z = point
    min_x, min_y, min_z, max_x, max_y, max_z = box_bounds
    return (min_x <= x <= max_x and min_y <= y <= max_y and min_z <= z <= max_z)