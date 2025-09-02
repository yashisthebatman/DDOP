# ==============================================================================
# File: ml_predictor/predictor.py
# ==============================================================================
import numpy as np
import os
import joblib
import logging

from config import (
    DRONE_SPEED_MPS, DRONE_VERTICAL_SPEED_MPS, DRONE_MASS_KG, DRONE_BASE_POWER_WATTS,
    DRONE_ADDITIONAL_WATTS_PER_KG, GRAVITY, ASCENT_EFFICIENCY, TURN_ENERGY_FACTOR
)
from utils.geometry import calculate_distance_3d, calculate_vector_angle_3d, calculate_wind_effect

class PhysicsBasedPredictor:
    """
    Physics-based model for energy and time prediction.
    This version includes a more realistic time model that differentiates
    between fast horizontal travel and slower vertical travel.
    """
    
    def predict(self, p1, p2, payload_kg, wind_vector, p_prev=None):
        p1, p2 = np.array(p1), np.array(p2)
        flight_vector = p2 - p1
        distance_3d = np.linalg.norm(flight_vector)
        
        if distance_3d < 1e-6:
            return 0, 0

        # --- Time Prediction ---
        horizontal_dist = np.linalg.norm([flight_vector[0], flight_vector[1]])
        vertical_dist = abs(flight_vector[2])

        horizontal_time = horizontal_dist / DRONE_SPEED_MPS
        vertical_time = vertical_dist / DRONE_VERTICAL_SPEED_MPS
        
        time_impact, energy_impact_factor = calculate_wind_effect(flight_vector, wind_vector, DRONE_SPEED_MPS)
        if time_impact == float('inf'):
            return float('inf'), float('inf')
        
        predicted_time = (horizontal_time * time_impact) + vertical_time
        
        # --- Energy Prediction ---
        total_mass_kg = DRONE_MASS_KG + payload_kg
        altitude_change = p2[2] - p1[2]
        potential_energy_wh = 0
        
        if altitude_change > 0:
            potential_energy_joules = (total_mass_kg * GRAVITY * altitude_change) / ASCENT_EFFICIENCY
            potential_energy_wh = potential_energy_joules / 3600

        # Energy to overcome drag and maintain lift is proportional to time spent flying
        base_power_to_hover = DRONE_BASE_POWER_WATTS + (total_mass_kg * DRONE_ADDITIONAL_WATTS_PER_KG) 
        horizontal_power = base_power_to_hover * energy_impact_factor
        horizontal_energy_wh = (horizontal_power * predicted_time) / 3600
        
        turning_energy_wh = 0
        if p_prev is not None:
            v1 = p1 - np.array(p_prev)
            v2 = flight_vector
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                angle_rad = calculate_vector_angle_3d(v1, v2)
                angle_deg = np.degrees(angle_rad)
                turning_energy_wh = TURN_ENERGY_FACTOR * angle_deg

        total_energy = potential_energy_wh + horizontal_energy_wh + turning_energy_wh
        return predicted_time, total_energy

class EnergyTimePredictor:
    """ML-powered predictor with a robust physics-based fallback."""
    
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
            features_2d = np.array(features).reshape(1, -1)
            time_pred = self.models['time_model'].predict(features_2d)[0]
            energy_pred = self.models['energy_model'].predict(features_2d)[0]
            return max(0, time_pred), max(0, energy_pred)
        except Exception as e:
            logging.warning(f"ML prediction failed: {e}. Using physics-based fallback.")
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
        
        wind_alignment = 0.0
        flight_vector_norm = np.linalg.norm(flight_vector)
        wind_vector_norm = np.linalg.norm(wind_vector)
        if flight_vector_norm > 1e-6 and wind_vector_norm > 1e-6:
            wind_alignment = np.dot(wind_vector, flight_vector) / (wind_vector_norm * flight_vector_norm)
        
        turning_angle = 0
        if p_prev is not None:
            v1 = np.array(p1) - np.array(p_prev)
            v2 = flight_vector
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                angle_rad = calculate_vector_angle_3d(v1, v2)
                turning_angle = np.degrees(angle_rad)
        
        return [
            distance_3d, altitude_change, horizontal_distance, payload_kg,
            wind_speed, wind_alignment, turning_angle, p1[2], p2[2], abs(altitude_change)
        ]