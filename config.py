# config.py
"""Central configuration file for the Q-DOP project."""

# --- Simulation Environment ---
AREA_BOUNDS = [-74.02, 40.70, -73.97, 40.74] # Bounding box for Lower Manhattan
HUB_LOCATION = (-74.013, 40.705, 100) # A hub near the southern tip
MIN_ALTITUDE = 100
MAX_ALTITUDE = 500 # Reduced max for urban environment
RECHARGE_TIME_S = 30 # 30 seconds to swap batteries/recharge at hub

# --- Drone & Physics ---
DRONE_SPEED_MPS = 25
DRONE_MAX_PAYLOAD_KG = 5.0
DRONE_BATTERY_WH = 20.0 # Reduced battery for more frequent returns
DRONE_MASS_KG = 2.0

# --- Default Scenario Parameters ---
DEFAULT_NUM_DRONES = 3

# --- ML Predictor ---
MODEL_PATH = 'ml_predictor/models/xgb_energy_time_model.json'

# --- QUBO Formulation ---
# For the new assignment problem, penalties will be handled differently