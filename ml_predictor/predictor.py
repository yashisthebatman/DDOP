# ml_predictor/predictor.py
import numpy as np
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import config
from utils.geometry import calculate_distance_3d, calculate_wind_effect, calculate_vector_angle_3d

GRAVITY = 9.81
ASCENT_EFFICIENCY = 0.7

class EnergyTimePredictor:
    """
    An advanced physics model for drone flight, incorporating inertial, rotational,
    and environmental effects.
    """
    def __init__(self):
        pass

    def calculate_inertial_energy(self, payload_kg):
        """Calculates energy for acceleration or deceleration."""
        total_mass = config.DRONE_MASS_KG + payload_kg
        return config.ACCELERATION_ENERGY_BASE_WH * total_mass

    def calculate_turning_energy(self, p_prev, p1, p2):
        """Calculates energy cost based on the angle of a turn."""
        if p_prev is None:
            return 0  # No turn at the start of a path
        
        v1 = np.array(p1) - np.array(p_prev)
        v2 = np.array(p2) - np.array(p1)
        
        angle = calculate_vector_angle_3d(v1, v2) # Angle in radians
        
        # Energy cost is proportional to how sharply the drone turns
        return config.TURN_ENERGY_FACTOR * angle

    def predict(self, p1, p2, payload_kg, wind_vector, p_prev=None):
        """
        Calculates time and energy for a single flight segment (p1 -> p2).
        Includes costs for cruise, altitude change, wind, and turning.
        """
        distance_3d = calculate_distance_3d(p1, p2)
        if distance_3d == 0: return 0, 0

        # --- Time Calculation (Affected by Wind) ---
        flight_vector = np.array(p2) - np.array(p1)
        time_impact, energy_impact_factor = calculate_wind_effect(flight_vector, wind_vector, config.DRONE_SPEED_MPS)
        
        if time_impact == float('inf'): # Drone cannot overcome headwind
            return float('inf'), float('inf')

        predicted_time = (distance_3d / config.DRONE_SPEED_MPS) * time_impact

        # --- Energy Calculation ---
        total_mass_kg = config.DRONE_MASS_KG + payload_kg
        
        # 1. Energy for altitude change (Potential Energy)
        altitude_change = p2[2] - p1[2]
        potential_energy_wh = 0
        if altitude_change > 0:
            potential_energy_joules = (total_mass_kg * GRAVITY * altitude_change) / ASCENT_EFFICIENCY
            potential_energy_wh = potential_energy_joules / 3600

        # 2. Energy for horizontal cruise flight (factoring in wind resistance)
        base_power = 50 + (total_mass_kg * 10) # Simplified power draw
        horizontal_power = base_power * energy_impact_factor
        horizontal_energy_wh = (horizontal_power * predicted_time) / 3600

        # 3. Energy for turning (Rotational Dynamics)
        turning_energy_wh = self.calculate_turning_energy(p_prev, p1, p2)

        total_energy = potential_energy_wh + horizontal_energy_wh + turning_energy_wh
        return predicted_time, total_energy