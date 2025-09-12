# FILE: ml_predictor/predictor.py

import numpy as np
import os
import joblib
import logging
import pandas as pd

from config import (
    DRONE_SPEED_MPS, DRONE_VERTICAL_SPEED_MPS, DRONE_MASS_KG, DRONE_BASE_POWER_WATTS,
    DRONE_ADDITIONAL_WATTS_PER_KG, GRAVITY, ASCENT_EFFICIENCY, TURN_ENERGY_FACTOR,
    MODEL_FILE_PATH
)
from utils.geometry import calculate_distance_3d, calculate_vector_angle_3d, calculate_wind_effect

class PhysicsBasedPredictor:
    # ... (This class is unchanged) ...
    def predict(self, p1, p2, payload_kg, wind_vector, p_prev=None):
        p1, p2 = np.array(p1), np.array(p2)
        flight_vector = p2 - p1
        if np.linalg.norm(flight_vector) < 1e-6:
            return 0, 0
        horizontal_dist = np.linalg.norm([flight_vector[0], flight_vector[1]])
        vertical_dist = abs(flight_vector[2])
        horizontal_time = horizontal_dist / DRONE_SPEED_MPS
        vertical_time = vertical_dist / DRONE_VERTICAL_SPEED_MPS
        time_impact, energy_impact_factor = calculate_wind_effect(flight_vector, wind_vector, DRONE_SPEED_MPS)
        if time_impact == float('inf'):
            return float('inf'), float('inf')
        predicted_time = (horizontal_time * time_impact) + vertical_time
        total_mass_kg = DRONE_MASS_KG + payload_kg
        altitude_change = p2[2] - p1[2]
        potential_energy_wh = 0
        if altitude_change > 0:
            potential_energy_joules = (total_mass_kg * GRAVITY * altitude_change) / ASCENT_EFFICIENCY
            potential_energy_wh = potential_energy_joules / 3600
        base_power_to_hover = DRONE_BASE_POWER_WATTS + (total_mass_kg * DRONE_ADDITIONAL_WATTS_PER_KG) 
        horizontal_power = base_power_to_hover * energy_impact_factor
        horizontal_energy_wh = (horizontal_power * predicted_time) / 3600
        turning_energy_wh = 0
        if p_prev is not None:
            v1 = p1 - np.array(p_prev)
            v2 = flight_vector
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                angle_rad = calculate_vector_angle_3d(v1, v2)
                turning_energy_wh = TURN_ENERGY_FACTOR * np.degrees(angle_rad)
        total_energy = potential_energy_wh + horizontal_energy_wh + turning_energy_wh
        return predicted_time, total_energy

class EnergyTimePredictor:
    """ML-powered predictor with a robust physics-based fallback."""
    def __init__(self):
        self.models = None
        self.fallback_predictor = PhysicsBasedPredictor()
        self.feature_names = [
            'distance_3d', 'altitude_change', 'horizontal_distance', 'payload_kg',
            'wind_speed', 'wind_alignment', 'turning_angle', 'p1_alt', 
            'p2_alt', 'abs_alt_change'
        ]
        # Model is no longer loaded on initialization

    def load_model(self, model_path=None):
        """Loads the model from the specified joblib file path."""
        if model_path is None:
            model_path = MODEL_FILE_PATH
        
        try:
            if os.path.exists(model_path):
                logging.info(f"Attempting to load ML model from '{model_path}'...")
                self.models = joblib.load(model_path)
                logging.info(f"✅ Successfully loaded ML model: {os.path.basename(model_path)}")
            else:
                logging.warning(f"⚠️ Model file not found at '{model_path}'. Using physics-based fallback only.")
                self.models = None
        except Exception as e:
            logging.error(f"❌ Error loading ML model from '{model_path}': {e}. Using physics-based fallback.")
            self.models = None

    def predict(self, p1, p2, payload_kg, wind_vector, p_prev=None):
        if not self.models:
            return self.fallback_predictor.predict(p1, p2, payload_kg, wind_vector, p_prev)
        try:
            features = self._extract_features(p1, p2, payload_kg, wind_vector, p_prev)
            features_df = pd.DataFrame([features], columns=self.feature_names)
            
            time_pred = self.models['time_model'].predict(features_df)[0]
            energy_pred = self.models['energy_model'].predict(features_df)[0]
            
            return max(0, time_pred), max(0, energy_pred)
        except Exception as e:
            logging.warning(f"Error during ML prediction: {e}. Using fallback.")
            return self.fallback_predictor.predict(p1, p2, payload_kg, wind_vector, p_prev)

    def _extract_features(self, p1, p2, payload_kg, wind_vector, p_prev=None):
        distance_3d = calculate_distance_3d(p1, p2)
        altitude_change = p2[2] - p1[2]
        horizontal_distance = np.linalg.norm([p2[0] - p1[0], p2[1] - p1[1]])
        wind_speed = np.linalg.norm(wind_vector)
        flight_vector = np.array(p2) - np.array(p1)
        wind_alignment = 0.0
        if np.linalg.norm(flight_vector) > 1e-6 and np.linalg.norm(wind_vector) > 1e-6:
            wind_alignment = np.dot(wind_vector, flight_vector) / (np.linalg.norm(wind_vector) * np.linalg.norm(flight_vector))
        turning_angle = 0
        if p_prev is not None:
            v1 = np.array(p1) - np.array(p_prev)
            if np.linalg.norm(v1) > 0 and np.linalg.norm(flight_vector) > 0:
                turning_angle = np.degrees(calculate_vector_angle_3d(v1, flight_vector))
        
        # This order must match self.feature_names and training data columns
        return {
            'distance_3d': distance_3d, 'altitude_change': altitude_change,
            'horizontal_distance': horizontal_distance, 'payload_kg': payload_kg,
            'wind_speed': wind_speed, 'wind_alignment': wind_alignment,
            'turning_angle': turning_angle, 'p1_alt': p1[2], 'p2_alt': p2[2],
            'abs_alt_change': abs(altitude_change)
        }