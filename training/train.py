# FILE: training/train.py

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging
import numpy as np
import joblib

from config import TRAINING_DATA_PATH

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" # Example local MLflow server
MODEL_NAME = "drone-energy-time-predictor"
EXPERIMENT_NAME = "Drone Mission Planners"

def run_training_on_dataframe(df, output_model_path):
    """
    Trains and evaluates the prediction models on a given DataFrame and saves the model.
    Returns success status and a dictionary of metrics.
    """
    features = [
        'distance_3d', 'altitude_change', 'horizontal_distance', 'payload_kg',
        'wind_speed', 'wind_alignment', 'turning_angle', 'p1_alt', 'p2_alt',
        'abs_alt_change'
    ]
    
    # Filter out rows where necessary columns might be missing
    required_cols = features + ['actual_time', 'actual_energy']
    df = df.dropna(subset=required_cols)
    
    X = df[features]
    y_time = df['actual_time']
    y_energy = df['actual_energy']
    
    X_train, X_test, y_time_train, y_time_test, y_energy_train, y_energy_test = train_test_split(
        X, y_time, y_energy, test_size=0.2, random_state=42
    )

    logging.info("Training time and energy models...")
    params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    
    time_model = RandomForestRegressor(**params).fit(X_train, y_time_train)
    energy_model = RandomForestRegressor(**params).fit(X_train, y_energy_train)

    logging.info("Evaluating models...")
    time_preds = time_model.predict(X_test)
    energy_preds = energy_model.predict(X_test)

    time_rmse = np.sqrt(mean_squared_error(y_time_test, time_preds))
    energy_rmse = np.sqrt(mean_squared_error(y_energy_test, energy_preds))

    metrics = {
        "time_rmse": time_rmse,
        "time_r2": r2_score(y_time_test, time_preds),
        "energy_rmse": energy_rmse,
        "energy_r2": r2_score(y_energy_test, energy_preds)
    }
    logging.info(f"Metrics: {metrics}")

    logging.info("Packaging and saving model locally...")
    model_package = {"time_model": time_model, "energy_model": energy_model}
    joblib.dump(model_package, output_model_path)
    
    # Optional: Log to MLflow if it's set up
    try:
        if mlflow.get_tracking_uri() is not None:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model=model_package,
                artifact_path="model",
                registered_model_name=MODEL_NAME
            )
            logging.info("Logged model to MLflow registry.")
    except Exception as e:
        logging.warning(f"Could not log to MLflow: {e}")

    return True, metrics

def main():
    """Main function for standalone training runs."""
    logging.basicConfig(level=logging.INFO)
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception as e:
        logging.warning(f"Could not connect to MLflow at {MLFLOW_TRACKING_URI}: {e}")

    with mlflow.start_run() as run:
        logging.info("Starting new training run...")
        
        try:
            df = pd.read_csv(TRAINING_DATA_PATH)
            logging.info(f"Loaded {len(df)} records from {TRAINING_DATA_PATH}")
        except FileNotFoundError:
            logging.error(f"Training data not found at {TRAINING_DATA_PATH}. Exiting.")
            return

        # Use the existing MODEL_FILE_PATH from config for standalone runs
        from config import MODEL_FILE_PATH
        run_training_on_dataframe(df, MODEL_FILE_PATH)
        logging.info("✅ Training run complete.")

if __name__ == "__main__":
    main()