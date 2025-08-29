# config.py
"""Central configuration file for the Q-DOP project."""

# --- Simulation Environment ---
AREA_BOUNDS = [0, 0, 10000, 10000]  # A 10km x 10km area
NUM_DRONES = 2
NUM_ORDERS = 5
NUM_BUILDINGS = 5 # How many 3D obstacles to generate
DRONE_SPEED_MPS = 15  # Meters per second
DRONE_MAX_PAYLOAD_KG = 5.0
DRONE_MAX_BATTERY_WH = 100.0
DRONE_BATTERY_RESERVE_FACTOR = 0.20

# --- ML Predictor ---
PAYLOAD_ENERGY_COEFFICIENT = 0.8  # Extra Wh consumed per kg per km
WIND_ENERGY_COEFFICIENT = 1.2   # Multiplier for energy consumption against headwind
MODEL_PATH = 'ml_predictor/models/xgb_energy_time_model.json'
TRAINING_DATA_PATH = 'ml_predictor/data/synthetic_flight_data.csv'

# --- QUBO Formulation ---
PENALTY_ORDER_NOT_VISITED = 10000
PENALTY_ONE_DRONE_PER_TIMESTEP = 10000
PENALTY_DRONE_ROUTE_CONTINUITY = 10000
PENALTY_BATTERY_EXCEEDED = 15000
PENALTY_PAYLOAD_EXCEEDED = 15000

# --- Objective Function Weights ---
TIME_WEIGHT = 0.7
ENERGY_WEIGHT = 0.3

# --- Solver Configuration ---
USE_DWAVE_SAMPLER = False
USE_OR_TOOLS_SOLVER = True