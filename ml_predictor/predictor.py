# ml_predictor/predictor.py
import numpy as np
import xgboost as xgb
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import config
from utils.geometry import calculate_distance_3d, calculate_wind_effect

# Physics constants for 3D energy calculation
GRAVITY = 9.81  # m/s^2
ASCENT_EFFICIENCY = 0.7  # Assume 70% motor efficiency when lifting

class EnergyTimePredictor:
    def __init__(self, model_path=config.MODEL_PATH):
        self.model = self._load_model(model_path)
        if self.model: print("INFO: XGBoost model loaded successfully.")
        else: print("WARN: Could not load XGBoost model. Falling back to dummy physics model.")

    def _load_model(self, model_path):
        if os.path.exists(model_path):
            model = xgb.XGBRegressor(); model.load_model(model_path); return model
        return None

    def _dummy_predict(self, p1, p2, payload_kg, wind_vector):
        """A 3D-aware physics model. Calculates cost based on distance AND altitude change."""
        distance_3d = calculate_distance_3d(p1, p2)
        if distance_3d == 0: return 0, 0
        
        flight_vector = np.array(p2) - np.array(p1)
        wind_effect_time, wind_effect_energy = calculate_wind_effect(flight_vector, wind_vector, config.DRONE_SPEED_MPS)
        
        altitude_change = p2[2] - p1[2]
        total_mass_kg = 2.0 + payload_kg # Assume drone mass is 2kg
        
        potential_energy_joules = 0
        if altitude_change > 0:
            potential_energy_joules = (total_mass_kg * GRAVITY * altitude_change) / ASCENT_EFFICIENCY
        
        potential_energy_wh = potential_energy_joules / 3600

        base_time = distance_3d / config.DRONE_SPEED_MPS
        predicted_time = base_time * wind_effect_time
        
        base_power_consumption = 50 
        payload_power = payload_kg * config.PAYLOAD_ENERGY_COEFFICIENT * 10
        horizontal_power = (base_power_consumption + payload_power) * wind_effect_energy
        horizontal_energy_wh = (horizontal_power * predicted_time) / 3600
        
        predicted_energy_wh = horizontal_energy_wh + potential_energy_wh
        return predicted_time, predicted_energy_wh

    def predict(self, p1, p2, payload_kg, wind_vector):
        return self._dummy_predict(p1, p2, payload_kg, wind_vector)

    def build_cost_matrices(self, locations, env, payload_kg=2.5):
        """Builds matrices considering 2D no-fly zones (but not 3D obstacles)."""
        num_locations = len(locations)
        time_matrix = np.full((num_locations, num_locations), np.inf)
        energy_matrix = np.full((num_locations, num_locations), np.inf)

        for i in range(num_locations):
            for j in range(num_locations):
                if i == j:
                    time_matrix[i, j], energy_matrix[i, j] = 0, 0
                    continue
                p1, p2 = locations[i], locations[j]
                
                if not env.is_path_valid(p1, p2): continue

                time, energy = self.predict(p1, p2, payload_kg, env.wind_vector)
                time_matrix[i, j], energy_matrix[i, j] = time, energy
        
        return time_matrix, energy_matrix