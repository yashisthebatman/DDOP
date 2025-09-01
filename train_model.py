# train_model.py
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import os

# Ensure the ml_predictor directory exists
os.makedirs("ml_predictor", exist_ok=True)

def train_model():
    """
    Loads synthetic data, trains separate XGBoost models for time and energy,
    and saves them together in a single file.
    """
    data_path = "ml_predictor/synthetic_drone_data.csv"
    model_path = "ml_predictor/drone_predictor_model.joblib"

    print("Loading data...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please run data_generator.py first to create the dataset.")
        return

    # Define features (X) and targets (y)
    features = [
        'distance_2d', 'altitude_change', 'payload_kg', 
        'wind_speed', 'wind_angle_deg', 'turn_angle_deg'
    ]
    targets = ['time_taken', 'energy_consumed']

    X = df[features]
    y = df[targets]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training XGBoost models for each target...")
    
    models = {}
    
    # Loop through each target and train a separate model
    for target in targets:
        print(f"--- Training for target: {target} ---")
        
        y_train_target = y_train[target]
        y_test_target = y_test[target]

        # Use XGBRegressor, which is highly performant
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,  # Use all available CPU cores
            early_stopping_rounds=50 # Stop if performance doesn't improve
        )

        # Train the model
        model.fit(
            X_train, y_train_target, 
            eval_set=[(X_test, y_test_target)], 
            verbose=False
        )

        models[target] = model
        print(f"Model for {target} trained successfully.")

    print("\nModel training complete. Evaluating performance...")
    
    # Evaluate the models
    y_pred_time = models['time_taken'].predict(X_test)
    y_pred_energy = models['energy_consumed'].predict(X_test)
    
    r2_time = r2_score(y_test['time_taken'], y_pred_time)
    mae_time = mean_absolute_error(y_test['time_taken'], y_pred_time)
    
    r2_energy = r2_score(y_test['energy_consumed'], y_pred_energy)
    mae_energy = mean_absolute_error(y_test['energy_consumed'], y_pred_energy)

    print("\n--- Model Performance ---")
    print(f"Time Prediction R²: {r2_time:.4f}")
    print(f"Time Prediction MAE: {mae_time:.4f} seconds")
    print(f"Energy Prediction R²: {r2_energy:.4f}")
    print(f"Energy Prediction MAE: {mae_energy:.4f} Wh")
    print("-------------------------\n")

    # Save the dictionary of trained models
    joblib.dump(models, model_path)
    print(f"Models saved successfully to {model_path}")

if __name__ == "__main__":
    train_model()