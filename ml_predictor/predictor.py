# ml_predictor.py
import numpy as np
import xgboost as xgb
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import config
from utils.geometry import calculate_distance_3d, calculate_wind_effect

GRAVITY = 9.81
ASCENT_EFFICIENCY = 0.7

class EnergyTimePredictor:
    def __init__(self, model_path=config.MODEL_PATH):
        # For this version, we rely purely on the physics model as it's more flexible
        # with dynamic payloads.
        pass

    def predict(self, p1, p2, payload_kg, wind_vector):
        """3D-aware physics model. Calculates cost based on distance AND altitude change."""
        distance_3d = calculate_distance_3d(p1, p2)
        if distance_3d == 0: return 0, 0
        
        flight_vector = np.array(p2) - np.array(p1)
        wind_effect_time, wind_effect_energy = calculate_wind_effect(flight_vector, wind_vector, config.DRONE_SPEED_MPS)
        
        altitude_change = p2[2] - p1[2]
        total_mass_kg = config.DRONE_MASS_KG + payload_kg
        
        potential_energy_joules = 0
        if altitude_change > 0:
            potential_energy_joules = (total_mass_kg * GRAVITY * altitude_change) / ASCENT_EFFICIENCY
        potential_energy_wh = potential_energy_joules / 3600

        base_time = distance_3d / config.DRONE_SPEED_MPS
        predicted_time = base_time * wind_effect_time
        
        base_power = 50 + (total_mass_kg * 10) # Simplified power draw based on mass
        horizontal_power = base_power * wind_effect_energy
        horizontal_energy_wh = (horizontal_power * predicted_time) / 3600
        
        return predicted_time, horizontal_energy_wh + potential_energy_wh