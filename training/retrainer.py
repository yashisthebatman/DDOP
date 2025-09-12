# FILE: training/retrainer.py

import pandas as pd
import logging
import os
import re
from datetime import datetime

# It's better to import the function rather than calling train.py as a subprocess
from training.train import run_training_on_dataframe
import system_state
from config import MODEL_FILE_PATH, TRAINING_DATA_PATH

logging.basicConfig(level=logging.INFO)

REAL_WORLD_DATA_PATH = 'data/real_world_flight_segments.csv'

def get_next_model_version_path(current_path):
    """Determines the next model version path, e.g., ..._v1.joblib -> ..._v2.joblib"""
    base, ext = os.path.splitext(current_path)
    match = re.search(r'_v(\d+)$', base)
    if match:
        version = int(match.group(1))
        next_version = version + 1
        new_base = base.rsplit('_v', 1)[0]
        return f"{new_base}_v{next_version}{ext}"
    else:
        # If no version tag, start with v2
        return f"{base}_v2{ext}"

def retrain_model():
    """
    Main function for the retraining feedback loop. Loads existing and new data,
    triggers a new training run, and updates the system state to point to the new model.
    """
    logging.info("--- Starting Model Retraining ---")
    
    # 1. Load Data
    try:
        initial_data = pd.read_csv(TRAINING_DATA_PATH)
        logging.info(f"Loaded {len(initial_data)} rows from initial training data.")
    except FileNotFoundError:
        logging.error(f"Initial training data not found at {TRAINING_DATA_PATH}. Aborting.")
        return False, "Initial training data not found."

    try:
        new_data = pd.read_csv(REAL_WORLD_DATA_PATH)
        logging.info(f"Loaded {len(new_data)} new rows from real-world operations.")
    except FileNotFoundError:
        logging.warning("No new real-world data found. Training on existing data only.")
        new_data = pd.DataFrame()

    if new_data.empty:
        logging.info("No new data to train on. Retraining process skipped.")
        return True, "No new data was available."

    combined_data = pd.concat([initial_data, new_data], ignore_index=True)
    logging.info(f"Total dataset size for retraining: {len(combined_data)} rows.")

    # 2. Determine New Model Path
    state = system_state.load_state()
    current_model_path = state.get('active_model_path', MODEL_FILE_PATH)
    new_model_path = get_next_model_version_path(current_model_path)
    logging.info(f"New model will be saved to: {new_model_path}")
    
    # 3. Run Training
    try:
        # This function should handle the actual ML training and model saving
        success, metrics = run_training_on_dataframe(combined_data, new_model_path)
        if not success:
            logging.error("Subprocess training script failed.")
            return False, "Training script failed to execute."
    except Exception as e:
        logging.error(f"An exception occurred during training: {e}")
        return False, str(e)

    # 4. Update State
    state['active_model_path'] = new_model_path
    state['log'].insert(0, f"🤖 Model retrained. New active model: {os.path.basename(new_model_path)}")
    system_state.save_state(state)
    logging.info(f"System state updated to use new model: {new_model_path}")
    
    return True, f"Model retrained successfully. Metrics: {metrics}"

if __name__ == "__main__":
    # Allows running this script manually
    retrain_model()