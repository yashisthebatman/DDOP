# ml_predictor/train_model.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import config
from utils.geometry import calculate_distance_3d, calculate_wind_effect

def generate_synthetic_data(num_samples=10000):
    """Generates a synthetic dataset for training the predictor model."""
    print("Generating synthetic flight data...")
    data = []
    for _ in range(num_samples):
        p1 = np.random.rand(3) * 1000
        p2 = np.random.rand(3) * 1000
        payload = np.random.uniform(0, config.DRONE_MAX_PAYLOAD_KG)
        wind_vector = np.random.randn(3) * 10 # Stronger, more varied wind
        
        distance = calculate_distance_3d(p1, p2)
        if distance < 1: continue
        
        flight_vector = p2 - p1
        wind_effect_time, wind_effect_energy = calculate_wind_effect(flight_vector, wind_vector, config.DRONE_SPEED_MPS)
        
        # Ground truth using our physics model
        base_time = distance / config.DRONE_SPEED_MPS
        true_time = base_time * wind_effect_time + np.random.normal(0, 5) # Add noise
        
        base_power = 50 + payload * config.PAYLOAD_ENERGY_COEFFICIENT * 10
        total_power = base_power * wind_effect_energy
        true_energy = (total_power * true_time) / 3600 + np.random.normal(0, 2) # Add noise

        # Features
        features = {
            'distance': distance,
            'payload_kg': payload,
            'wind_x': wind_vector[0],
            'wind_y': wind_vector[1],
            'flight_vec_x': flight_vector[0],
            'flight_vec_y': flight_vector[1],
            'wind_effect_time': wind_effect_time,
            'wind_effect_energy': wind_effect_energy,
            'true_time': true_time,
            'true_energy': true_energy
        }
        data.append(features)
        
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(config.TRAINING_DATA_PATH), exist_ok=True)
    df.to_csv(config.TRAINING_DATA_PATH, index=False)
    print(f"Synthetic data saved to {config.TRAINING_DATA_PATH}")
    return df

def train():
    """Trains the XGBoost model and saves it."""
    if not os.path.exists(config.TRAINING_DATA_PATH):
        df = generate_synthetic_data()
    else:
        print(f"Loading existing data from {config.TRAINING_DATA_PATH}")
        df = pd.read_csv(config.TRAINING_DATA_PATH)

    df = df.dropna()
    df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]

    features = ['distance', 'payload_kg', 'wind_x', 'wind_y', 'flight_vec_x', 'flight_vec_y', 'wind_effect_time', 'wind_effect_energy']
    targets = ['true_time', 'true_energy']
    
    X = df[features]
    y = df[targets]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost model...")
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, multi_strategy='multi_output_tree', n_jobs=-1)
    
    # --- CHANGE IS HERE ---
    # The old way (deprecated):
    # model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    
    # The new, correct way using callbacks for XGBoost >= 2.0
    early_stopping = xgb.callback.EarlyStopping(rounds=10, save_best=True)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[early_stopping], verbose=False)
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    model.save_model(config.MODEL_PATH)
    print(f"Model trained and saved to {config.MODEL_PATH}")

if __name__ == '__main__':
    train()