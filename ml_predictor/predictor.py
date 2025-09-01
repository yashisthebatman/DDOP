# ml_predictor/predictor.py (Full and Corrected)

import logging
import os
import sys
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path to allow for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import config
from utils.geometry import calculate_distance_3d, calculate_vector_angle_3d, calculate_wind_effect

class EnergyTimePredictor:
    """
    An ML-powered predictor. It loads trained XGBoost models for time and energy,
    using a physics-based model as a fallback if the ML models are not found.
    """
    def __init__(self, model_path="ml_predictor/drone_predictor_model.joblib"):
        self.models = None
        try:
            self.models = joblib.load(model_path)
            logging.info(f"✅ Successfully loaded ML models from {model_path}")
        except FileNotFoundError:
            logging.warning(f"⚠️ ML model not found at {model_path}. Using physics-based fallback.")
        except Exception as e:
            logging.error(f"❌ Error loading ML model: {e}. Using physics-based fallback.")

    def predict(self, p1: np.ndarray, p2: np.ndarray, payload_kg: float, wind_vector: np.ndarray = np.array([0,0,0]), p_prev: np.ndarray = None):
        """
        CORRECTED: The method name is now 'predict' to match calls from the planner.
        Predicts the time (in seconds) and energy (in Wh) for a flight segment.
        """
        distance_3d = np.linalg.norm(p2 - p1)
        if distance_3d < 1e-6:
            return 0.0, 0.0

        # Fallback to physics model if ML models are not loaded
        if not self.models:
            return self._physics_predict(p1, p2, payload_kg, wind_vector, p_prev)

        try:
            # --- Feature Engineering ---
            flight_vector = p2 - p1
            distance_2d = np.linalg.norm(flight_vector[:2])
            altitude_change = flight_vector[2]

            wind_speed = np.linalg.norm(wind_vector)
            if wind_speed > 0 and distance_2d > 0:
                flight_dir_2d = flight_vector[:2] / distance_2d
                wind_dir_2d = wind_vector[:2] / wind_speed
                dot_product = np.clip(np.dot(flight_dir_2d, wind_dir_2d), -1.0, 1.0)
                wind_angle_deg = np.degrees(np.arccos(dot_product))
            else:
                wind_angle_deg = 0.0

            turn_angle_deg = 0.0
            if p_prev is not None:
                v1 = p1 - p_prev
                v2 = p2 - p1
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    turn_angle_deg = np.degrees(calculate_vector_angle_3d(v1, v2))

            # --- Prediction ---
            features = pd.DataFrame([{
                'distance_2d': distance_2d,
                'altitude_change': altitude_change,
                'payload_kg': payload_kg,
                'wind_speed': wind_speed,
                'wind_angle_deg': wind_angle_deg,
                'turn_angle_deg': turn_angle_deg
            }])
            
            pred_time = self.models['time_taken'].predict(features)[0]
            pred_energy = self.models['energy_consumed'].predict(features)[0]

            return max(0, pred_time), max(0, pred_energy)

        except Exception as e:
            logging.error(f"Error during ML prediction: {e}. Falling back to physics model.")
            return self._physics_predict(p1, p2, payload_kg, wind_vector, p_prev)

    def _physics_predict(self, p1, p2, payload_kg, wind_vector, p_prev=None):
        """The original physics model, used as a fallback."""
        distance_3d = calculate_distance_3d(p1, p2)
        if distance_3d == 0: return 0, 0
        flight_vector = np.array(p2) - np.array(p1)

        time_impact, energy_impact_factor = calculate_wind_effect(flight_vector, wind_vector, config.DRONE_SPEED_MPS)
        
        if time_impact == float('inf'): return float('inf'), float('inf')
        predicted_time = (distance_3d / config.DRONE_SPEED_MPS) * time_impact

        total_mass_kg = config.DRONE_MASS_KG + payload_kg
        altitude_change = p2[2] - p1[2]
        potential_energy_wh = 0
        if altitude_change > 0:
            potential_energy_joules = (total_mass_kg * config.GRAVITY * altitude_change) / config.ASCENT_EFFICIENCY
            potential_energy_wh = potential_energy_joules / 3600

        base_power = 50 + (total_mass_kg * 10) # Simplified power model
        horizontal_power = base_power * energy_impact_factor
        horizontal_energy_wh = (horizontal_power * predicted_time) / 3600
        
        turning_energy_wh = 0
        if p_prev is not None:
            v1 = np.array(p1) - np.array(p_prev)
            v2 = np.array(p2) - np.array(p1)
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                angle = np.degrees(calculate_vector_angle_3d(v1, v2))
                turning_energy_wh = config.TURN_ENERGY_FACTOR * angle

        total_energy = potential_energy_wh + horizontal_energy_wh + turning_energy_wh
        return predicted_time, total_energy