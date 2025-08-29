# ml_predictor/predictor.py
import numpy as np
import xgboost as xgb
import os
import sys

# Add project root to path for robust imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import config
from utils.geometry import calculate_distance_3d, calculate_wind_effect

class EnergyTimePredictor:
    """Predicts time and energy for a drone flight segment."""
    def __init__(self, model_path=config.MODEL_PATH):
        self.model = self._load_model(model_path)
        if self.model:
            print("INFO: XGBoost model loaded successfully.")
        else:
            print("WARN: Could not load XGBoost model. Falling back to dummy physics model.")

    def _load_model(self, model_path):
        if os.path.exists(model_path):
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            return model
        return None

    def _dummy_predict(self, distance, payload, wind_effect_time, wind_effect_energy):
        """A simplified physics model as a fallback."""
        base_time = distance / config.DRONE_SPEED_MPS
        predicted_time = base_time * wind_effect_time
        
        base_power_consumption = 50
        payload_power = payload * config.PAYLOAD_ENERGY_COEFFICIENT * 10
        total_power = (base_power_consumption + payload_power) * wind_effect_energy
        predicted_energy_wh = (total_power * predicted_time) / 3600
        
        return predicted_time, predicted_energy_wh

    def predict(self, p1, p2, payload_kg, wind_vector):
        """Predicts time and energy for a single flight segment."""
        distance = calculate_distance_3d(p1, p2)
        if distance == 0:
            return 0, 0
            
        flight_vector = np.array(p2) - np.array(p1)
        wind_effect_time, wind_effect_energy = calculate_wind_effect(flight_vector, wind_vector, config.DRONE_SPEED_MPS)

        if self.model:
            features = np.array([[
                distance, payload_kg, wind_vector[0], wind_vector[1],
                flight_vector[0], flight_vector[1], wind_effect_time, wind_effect_energy
            ]])
            prediction = self.model.predict(features)[0]
            return prediction[0], prediction[1]
        else:
            return self._dummy_predict(distance, payload_kg, wind_effect_time, wind_effect_energy)

    def build_cost_matrices(self, locations, env, payload_kg=2.5):
        """
        Builds matrices of predicted time and energy for all-to-all location pairs.
        
        CHANGE: Now accepts the 'env' object to check for no-fly zones.
        Paths that cross no-fly zones are given an infinite cost.
        """
        num_locations = len(locations)
        time_matrix = np.full((num_locations, num_locations), np.inf)
        energy_matrix = np.full((num_locations, num_locations), np.inf)

        for i in range(num_locations):
            for j in range(num_locations):
                if i == j:
                    time_matrix[i, j] = 0
                    energy_matrix[i, j] = 0
                    continue
                
                p1 = locations[i]
                p2 = locations[j]
                
                # LOGIC CHANGE: Check if path is valid before calculating cost
                if not env.is_path_valid(p1, p2):
                    # Cost remains infinite, optimizer will avoid this path
                    continue

                time, energy = self.predict(p1, p2, payload_kg, env.wind_vector)
                time_matrix[i, j] = time
                energy_matrix[i, j] = energy
        
        return time_matrix, energy_matrix