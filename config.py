# config.py
"""Central configuration file for the Q-DOP project."""

# --- Simulation Environment ---
AREA_BOUNDS = [0, 0, 10000, 10000]
MIN_ALTITUDE = 100  # Minimum flight altitude for the drone
MAX_ALTITUDE = 600  # Maximum flight altitude for the drone
DRONE_SPEED_MPS = 20  # Increased speed for larger scenarios
DRONE_MAX_PAYLOAD_KG = 5.0
DRONE_MAX_BATTERY_WH = 100.0

# --- Default Scenario Parameters (can be overridden by UI) ---
DEFAULT_NUM_DRONES = 2
DEFAULT_NUM_ORDERS = 8
DEFAULT_NUM_BUILDINGS = 15

# --- ML Predictor ---
PAYLOAD_ENERGY_COEFFICIENT = 0.8
WIND_ENERGY_COEFFICIENT = 1.2
MODEL_PATH = 'ml_predictor/models/xgb_energy_time_model.json'
TRAINING_DATA_PATH = 'ml_predictor/data/synthetic_flight_data.csv'

# --- QUBO Formulation ---
PENALTY_ORDER_NOT_VISITED = 15000  # Increased penalty
PENALTY_ONE_DRONE_PER_TIMESTEP = 10000
PENALTY_DRONE_ROUTE_CONTINUITY = 10000

# --- Solver Configuration ---
USE_DWAVE_SAMPLER = False
USE_OR_TOOLS_SOLVER = True