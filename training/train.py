import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging
import numpy as np # Import numpy for sqrt

# --- Configuration ---
# In a real app, this would come from a config file or environment variables
DB_CONNECTION_STRING = "postgresql://user:password@host:port/database"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" # Example local MLflow server
MODEL_NAME = "drone-energy-time-predictor"
EXPERIMENT_NAME = "Drone Mission Planners"

def load_data_from_db():
    """
    Placeholder function to load flight leg data from a database.
    In a real scenario, you'd use SQLAlchemy or a similar library.
    """
    logging.info(f"Loading data from {DB_CONNECTION_STRING}...")
    # This is mock data. Replace with actual DB query.
    data = {
        'distance_3d': [100, 150, 200, 50, 300],
        'altitude_change': [10, -5, 20, 5, -15],
        'horizontal_distance': [99.5, 149.9, 199.0, 49.7, 299.6],
        'payload_kg': [1.0, 2.5, 0.5, 5.0, 1.5],
        'wind_speed': [2.0, 5.0, 1.0, 8.0, 3.0],
        'wind_alignment': [0.5, -0.8, 0.9, -0.1, 0.0],
        'turning_angle': [0, 90, 15, 45, 180],
        'start_altitude': [50, 60, 40, 80, 100],
        'end_altitude': [60, 55, 60, 85, 85],
        'abs_altitude_change': [10, 5, 20, 5, 15],
        'actual_time': [10.2, 18.1, 15.5, 8.0, 25.1],
        'actual_energy': [5.5, 12.3, 8.1, 9.5, 15.2]
    }
    return pd.DataFrame(data)

def main():
    logging.basicConfig(level=logging.INFO)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        logging.info("Starting new training run...")
        
        # 1. Load Data
        df = load_data_from_db()
        
        features = [col for col in df.columns if col not in ['actual_time', 'actual_energy']]
        X = df[features]
        y_time = df['actual_time']
        y_energy = df['actual_energy']
        
        X_train, X_test, y_time_train, y_time_test, y_energy_train, y_energy_test = train_test_split(
            X, y_time, y_energy, test_size=0.2, random_state=42
        )

        # 2. Train Models
        logging.info("Training time and energy models...")
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        mlflow.log_params(params)
        
        time_model = RandomForestRegressor(**params).fit(X_train, y_time_train)
        energy_model = RandomForestRegressor(**params).fit(X_train, y_energy_train)

        # 3. Evaluate Models
        logging.info("Evaluating models...")
        time_preds = time_model.predict(X_test)
        energy_preds = energy_model.predict(X_test)

        # FIX: The 'squared' argument for mean_squared_error requires scikit-learn >= 0.22.
        # This implementation is backward-compatible with older versions.
        time_mse = mean_squared_error(y_time_test, time_preds)
        energy_mse = mean_squared_error(y_energy_test, energy_preds)

        metrics = {
            "time_rmse": np.sqrt(time_mse),
            "time_r2": r2_score(y_time_test, time_preds),
            "energy_rmse": np.sqrt(energy_mse),
            "energy_r2": r2_score(y_energy_test, energy_preds)
        }
        mlflow.log_metrics(metrics)
        logging.info(f"Metrics: {metrics}")

        # 4. Package and Log Model to Registry
        logging.info("Logging model to MLflow Registry...")
        model_package = {"time_model": time_model, "energy_model": energy_model}
        
        mlflow.sklearn.log_model(
            sk_model=model_package,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )
        logging.info("✅ Training run complete.")

if __name__ == "__main__":
    main()