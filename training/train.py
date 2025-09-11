import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging
import numpy as np

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
        'distance_3d': np.random.uniform(50, 500, 100),
        'altitude_change': np.random.uniform(-50, 50, 100),
        'horizontal_distance': np.random.uniform(50, 500, 100),
        'payload_kg': np.random.uniform(0.1, 5.0, 100),
        'wind_speed': np.random.uniform(0, 10, 100),
        'wind_alignment': np.random.uniform(-1, 1, 100),
        'turning_angle': np.random.uniform(0, 180, 100),
        'start_altitude': np.random.uniform(20, 150, 100),
        'end_altitude': np.random.uniform(20, 150, 100),
        'abs_altitude_change': np.random.uniform(0, 50, 100),
        'actual_time': np.random.uniform(10, 60, 100),
        'actual_energy': np.random.uniform(5, 25, 100)
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

        # The 'squared=False' argument for mean_squared_error is not universally
        # available in all scikit-learn versions. This method is backward-compatible.
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