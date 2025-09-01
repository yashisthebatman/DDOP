# ml_predictor/predictor.py
import numpy as np
import os, sys
import joblib
import logging
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import config
from utils.geometry import calculate_distance_3d, calculate_vector_angle_3d

class PhysicsBasedPredictor:
    # ... (This class is unchanged) ...
    """The original physics model, used for data generation and as a fallback."""
    def predict(self, p1, p2, payload_kg, wind_vector, p_prev=None):
        distance_3d = calculate_distance_3d(p1, p2)
        if distance_3d == 0: return 0, 0
        flight_vector = np.array(p2) - np.array(p1)

        # This import is moved here to avoid circular dependencies
        from utils.geometry import calculate_wind_effect
        time_impact, energy_impact_factor = calculate_wind_effect(flight_vector, wind_vector, config.DRONE_SPEED_MPS)
        
        if time_impact == float('inf'): return float('inf'), float('inf')
        predicted_time = (distance_3d / config.DRONE_SPEED_MPS) * time_impact

        total_mass_kg = config.DRONE_MASS_KG + payload_kg
        altitude_change = p2[2] - p1[2]
        potential_energy_wh = 0
        if altitude_change > 0:
            potential_energy_joules = (total_mass_kg * config.GRAVITY * altitude_change) / config.ASCENT_EFFICIENCY
            potential_energy_wh = potential_energy_joules / 3600

        base_power = 50 + (total_mass_kg * 10)
        horizontal_power = base_power * energy_impact_factor
        horizontal_energy_wh = (horizontal_power * predicted_time) / 3600
        
        turning_energy_wh = 0
        if p_prev is not None:
            v1 = np.array(p1) - np.array(p_prev); v2 = np.array(p2) - np.array(p1)
            angle = calculate_vector_angle_3d(v1, v2)
            turning_energy_wh = config.TURN_ENERGY_FACTOR * angle

        total_energy = potential_energy_wh + horizontal_energy_wh + turning_energy_wh
        return predicted_time, total_energy

class EnergyTimePredictor:
    """
    An ML-powered predictor. It loads a dictionary of trained XGBoost models and uses a
    physics-based model as a fallback if the ML models are not found.
    """
    def __init__(self, model_path="ml_predictor/drone_predictor_model.joblib"):
        self.models = None # Changed from self.model to self.models
        self.fallback_predictor = PhysicsBasedPredictor()
        try:
            self.models = joblib.load(model_path)
            logging.info(f"✅ Successfully loaded ML models from {model_path}")
        except FileNotFoundError:
            logging.warning(f"⚠️ ML model not found at {model_path}. Using physics-based fallback.")
        except Exception as e:
            logging.error(f"❌ Error loading ML model: {e}. Using physics-based fallback.")

    def predict(self, p1, p2, payload_kg, wind_vector, p_prev=None):
        if not self.models:
            return self.fallback_predictor.predict(p1, p2, payload_kg, wind_vector, p_prev)

        try:
            # --- Feature Engineering ---
            p1_np, p2_np = np.array(p1), np.array(p2)
            flight_vector = p2_np - p1_np
            distance_3d = np.linalg.norm(flight_vector)
            if distance_3d < 1e-6: return 0.0, 0.0
            
            distance_2d = np.linalg.norm(flight_vector[:2])
            altitude_change = flight_vector[2]

            wind_speed = np.linalg.norm(wind_vector)
            flight_dir_2d = flight_vector[:2] / distance_2d if distance_2d > 0 else np.array([1, 0])
            wind_dir_2d = wind_vector[:2] / wind_speed if wind_speed > 0 else np.array([1, 0])
            
            dot_product = np.clip(np.dot(flight_dir_2d, wind_dir_2d), -1.0, 1.0)
            wind_angle_rad = np.arccos(dot_product)
            wind_angle_deg = np.degrees(wind_angle_rad)

            turn_angle_deg = 0.0
            if p_prev:
                v1 = p1_np - np.array(p_prev)
                v2 = p2_np - p1_np
                turn_angle_rad = calculate_vector_angle_3d(v1, v2)
                turn_angle_deg = np.degrees(turn_angle_rad)

            # --- Prediction ---
            features = pd.DataFrame([{
                'distance_2d': distance_2d,
                'altitude_change': altitude_change,
                'payload_kg': payload_kg,
                'wind_speed': wind_speed,
                'wind_angle_deg': wind_angle_deg,
                'turn_angle_deg': turn_angle_deg
            }])
            
            # Predict each target separately using the corresponding model
            pred_time = self.models['time_taken'].predict(features)[0]
            pred_energy = self.models['energy_consumed'].predict(features)[0]

            # Ensure predictions are non-negative
            return max(0, pred_time), max(0, pred_energy)

        except Exception as e:
            logging.error(f"Error during ML prediction: {e}. Falling back to physics model for this segment.")
            return self.fallback_predictor.predict(p1, p2, payload_kg, wind_vector, p_prev)