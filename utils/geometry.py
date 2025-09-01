import numpy as np

def calculate_distance_3d(p1, p2):
    """Calculates the Euclidean distance between two 3D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_vector_angle_3d(v1, v2):
    """Calculates the angle in radians between two 3D vectors."""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    # Clip the dot product to handle potential floating point inaccuracies
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot_product)

def calculate_wind_effect(flight_vector, wind_vector, drone_speed):
    """Returns a tuple of (time_impact_factor, energy_impact_factor)."""
    norm_flight_vector = np.linalg.norm(flight_vector)
    if norm_flight_vector == 0: return 1.0, 1.0

    flight_unit_vector = flight_vector / norm_flight_vector
    wind_component = np.dot(wind_vector, flight_unit_vector)
    
    effective_speed = drone_speed + wind_component

    if effective_speed <= 1e-6:
        return float('inf'), float('inf')

    time_impact_factor = drone_speed / effective_speed
    energy_impact_factor = (drone_speed / effective_speed)**2

    return time_impact_factor, energy_impact_factor

def line_segment_intersects_aabb(p1: tuple, p2: tuple, box_bounds: tuple) -> bool:
    """
    Efficiently checks if a 3D line segment intersects with an Axis-Aligned Bounding Box (AABB).
    Uses the slab test method (Kay/Kajiya algorithm).
    
    Args:
        p1: The starting point of the line segment (x, y, z).
        p2: The ending point of the line segment (x, y, z).
        box_bounds: The AABB bounds (min_x, min_y, min_z, max_x, max_y, max_z).
    
    Returns:
        True if the segment intersects the box, False otherwise.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    min_b = np.array(box_bounds[:3])
    max_b = np.array(box_bounds[3:])
    
    # Segment's direction vector
    direction = p2 - p1
    
    # Temporarily suppress the "invalid value" warning for the 0*inf edge case
    with np.errstate(invalid='ignore'):
        # Avoid division by zero for axis-aligned lines
        dir_inv = np.divide(1.0, direction, out=np.full_like(direction, np.inf), where=direction!=0)
        
        t1 = (min_b - p1) * dir_inv
        t2 = (max_b - p1) * dir_inv
    
    tmin = np.max(np.minimum(t1, t2))
    tmax = np.min(np.maximum(t1, t2))

    # The segment intersects the box if tmax is greater than tmin,
    # and the intersection lies within the segment (tmax > 0 and tmin < 1).
    return tmax > max(0, tmin) and tmin < 1