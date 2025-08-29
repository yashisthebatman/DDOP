# utils/geometry.py
import numpy as np

def calculate_distance_3d(p1, p2):
    """Calculates the Euclidean distance between two 3D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_wind_effect(flight_vector, wind_vector, drone_speed):
    """
    Calculates the headwind/tailwind component.
    
    Returns a tuple of (time_impact_factor, energy_impact_factor).
    """
    # Handle zero-length flight vector to avoid division by zero
    norm_flight_vector = np.linalg.norm(flight_vector)
    if norm_flight_vector == 0:
        return 1.0, 1.0

    flight_unit_vector = flight_vector / norm_flight_vector
    # Project wind vector onto the flight path
    wind_component = np.dot(wind_vector, flight_unit_vector)
    
    # Effective speed is drone speed minus the headwind component
    effective_speed = drone_speed - wind_component
    
    # Avoid division by zero or negative speed
    if effective_speed <= 1e-6: # Use a small epsilon for float comparison
        # CHANGE: Always return a tuple, even for an impassable route.
        return float('inf'), float('inf')
        
    # The time taken is proportional to (speed / effective_speed)
    # The energy factor is proportional to (speed / effective_speed)^2 (simplified physics)
    time_impact_factor = drone_speed / effective_speed
    energy_impact_factor = (drone_speed / effective_speed)**2
    
    return time_impact_factor, energy_impact_factor