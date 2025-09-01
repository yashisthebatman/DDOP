# data_generator.py
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os

import config
from ml_predictor.predictor import PhysicsBasedPredictor
from utils.geometry import calculate_vector_angle_3d

# Ensure the ml_predictor directory exists
os.makedirs("ml_predictor", exist_ok=True)

def generate_synthetic_data(num_samples=75000):
    """
    Generates a synthetic dataset for training a drone flight predictor model.
    """
    print(f"Generating {num_samples} synthetic flight data samples...")
    
    predictor = PhysicsBasedPredictor()
    data = []

    for _ in tqdm(range(num_samples)):
        # 1. Randomize flight segment properties
        distance_2d = random.uniform(20, 2000)
        start_altitude = random.uniform(config.MIN_ALTITUDE, config.MAX_ALTITUDE)
        end_altitude = random.uniform(config.MIN_ALTITUDE, config.MAX_ALTITUDE)
        
        # Create 3D points based on these properties
        p1 = (0, 0, start_altitude)
        altitude_change = end_altitude - start_altitude
        p2 = (distance_2d, 0, end_altitude)

        # 2. Randomize payload
        payload_kg = random.uniform(0.0, config.DRONE_MAX_PAYLOAD_KG)

        # 3. Randomize wind conditions
        wind_speed = random.uniform(0, 25)
        # Angle of wind relative to flight path (0=tailwind, 180=headwind)
        wind_angle_deg = random.uniform(0, 360)
        wind_angle_rad = np.radians(wind_angle_deg)
        wind_vector = np.array([np.cos(wind_angle_rad) * wind_speed, np.sin(wind_angle_rad) * wind_speed, 0])

        # 4. Randomize turn angle
        turn_angle_deg = random.uniform(0, 180)
        # To simulate this, create a p_prev that results in the desired angle
        if turn_angle_deg > 1:
            # Create a previous point to simulate a turn
            p_prev = (-100, np.tan(np.radians(180-turn_angle_deg)) * 100, start_altitude)
        else:
            p_prev = None # Straight flight

        # 5. Calculate the outcome using the physics model
        time_taken, energy_consumed = predictor.predict(p1, p2, payload_kg, wind_vector, p_prev)

        # 6. Store the results
        if time_taken != float('inf'):
            data.append({
                'distance_2d': distance_2d,
                'altitude_change': altitude_change,
                'payload_kg': payload_kg,
                'wind_speed': wind_speed,
                'wind_angle_deg': wind_angle_deg,
                'turn_angle_deg': turn_angle_deg,
                'time_taken': time_taken,
                'energy_consumed': energy_consumed,
            })

    df = pd.DataFrame(data)
    output_path = "ml_predictor/synthetic_drone_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Data generation complete. Saved to {output_path}")

if __name__ == "__main__":
    generate_synthetic_data()