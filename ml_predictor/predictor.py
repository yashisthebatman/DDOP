import numpy as np
import os
import sys
import joblib
import logging
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import config
from utils.geometry import calculate_distance_3d, calculate_vector_angle_3d, calculate_wind_effect

class PhysicsBasedPredictor:
    """Physics-based model for energy and time prediction."""
    
    def predict(self, p1, p2, payload_kg, wind_vector, p_prev=None):
        distance_3d = calculate_distance_3d(p1, p2)
        if distance_3d == 0:
            return 0, 0
        
        flight_vector = np.array(p2) - np.array(p1)
        time_impact, energy_impact_factor = calculate_wind_effect(flight_vector, wind_vector, config.DRONE_SPEED_MPS)
        
        if time_impact == float('inf'):
            return float('inf'), float('inf')
        
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
            v1 = np.array(p1) - np.array(p_prev)
            v2 = np.array(p2) - np.array(p1)
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                angle = calculate_vector_angle_3d(v1, v2)
                turning_energy_wh = config.TURN_ENERGY_FACTOR * angle

        total_energy = potential_energy_wh + horizontal_energy_wh + turning_energy_wh
        return predicted_time, total_energy

class EnergyTimePredictor:
    """ML-powered predictor with physics fallback."""
    
    def __init__(self, model_path="ml_predictor/drone_predictor_model.joblib"):
        self.models = None
        self.fallback_predictor = PhysicsBasedPredictor()
        
        try:
            if os.path.exists(model_path):
                self.models = joblib.load(model_path)
                logging.info(f"✅ Successfully loaded ML models from {model_path}")
            else:
                logging.warning(f"⚠️ ML model not found at {model_path}. Using physics-based fallback.")
        except Exception as e:
            logging.error(f"❌ Error loading ML model: {e}. Using physics-based fallback.")

    def predict(self, p1, p2, payload_kg, wind_vector, p_prev=None):
        if not self.models:
            return self.fallback_predictor.predict(p1, p2, payload_kg, wind_vector, p_prev)

        try:
            features = self._extract_features(p1, p2, payload_kg, wind_vector, p_prev)
            time_pred = self.models['time_model'].predict([features])[0]
            energy_pred = self.models['energy_model'].predict([features])[0]
            return max(0, time_pred), max(0, energy_pred)
        except Exception as e:
            logging.warning(f"ML prediction failed: {e}. Using fallback.")
            return self.fallback_predictor.predict(p1, p2, payload_kg, wind_vector, p_prev)

    def predict_energy_time(self, p1, p2, payload_kg, wind_vector=None, p_prev=None):
        """Wrapper method for compatibility."""
        if wind_vector is None:
            wind_vector = np.array([0, 0, 0])
        return self.predict(p1, p2, payload_kg, wind_vector, p_prev)

    def _extract_features(self, p1, p2, payload_kg, wind_vector, p_prev=None):
        """Extract features for ML prediction."""
        distance_3d = calculate_distance_3d(p1, p2)
        altitude_change = p2[2] - p1[2]
        horizontal_distance = np.linalg.norm([p2[0] - p1[0], p2[1] - p1[1]])
        
        wind_speed = np.linalg.norm(wind_vector)
        flight_vector = np.array(p2) - np.array(p1)
        if np.linalg.norm(flight_vector) > 0:
            wind_alignment = np.dot(wind_vector, flight_vector) / (np.linalg.norm(wind_vector) * np.linalg.norm(flight_vector))
        else:
            wind_alignment = 0
        
        turning_angle = 0
        if p_prev is not None:
            v1 = np.array(p1) - np.array(p_prev)
            v2 = np.array(p2) - np.array(p1)
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                turning_angle = calculate_vector_angle_3d(v1, v2)
        
        return [
            distance_3d, altitude_change, horizontal_distance, payload_kg,
            wind_speed, wind_alignment, turning_angle, p1[2], p2[2], abs(altitude_change)
        ]