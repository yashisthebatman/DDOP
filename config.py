# config.py
"""Central configuration file for the Q-DOP project."""

# --- Simulation Environment ---
# Operational area defined as a bounding box [min_lat, min_lon, max_lat, max_lon]
AREA_BOUNDS = [0, 0, 10000, 10000]  # A 10km x 10km area
NUM_DRONES = 2
NUM_ORDERS = 5
NUM_CHARGING_PADS = 2
DRONE_SPEED_MPS = 15  # Meters per second
DRONE_MAX_PAYLOAD_KG = 5.0
# Battery in Watt-hours. A 100Wh battery can provide 100W for 1 hour.
DRONE_MAX_BATTERY_WH = 100.0
# Keep 20% in reserve
DRONE_BATTERY_RESERVE_FACTOR = 0.20

# --- ML Predictor ---
# Weights for the dummy physics-based cost model
# In a real scenario, these would be learned by the ML model
PAYLOAD_ENERGY_COEFFICIENT = 0.8  # Extra Wh consumed per kg per km
WIND_ENERGY_COEFFICIENT = 1.2   # Multiplier for energy consumption against headwind
MODEL_PATH = 'ml_predictor/models/xgb_energy_time_model.json'
TRAINING_DATA_PATH = 'ml_predictor/data/synthetic_flight_data.csv'

# --- QUBO Formulation ---
# Penalty values must be tuned. They should be larger than any possible objective value.
PENALTY_ORDER_NOT_VISITED = 10000
PENALTY_ONE_DRONE_PER_TIMESTEP = 10000
PENALTY_DRONE_ROUTE_CONTINUITY = 10000
PENALTY_BATTERY_EXCEEDED = 15000
PENALTY_PAYLOAD_EXCEEDED = 15000

# Weights for the objective function
# How much to prioritize time vs. energy. 0.7 means we care more about time.
TIME_WEIGHT = 0.7
ENERGY_WEIGHT = 0.3

# --- Solver Configuration ---
USE_DWAVE_SAMPLER = False  # Set to True if you have a D-Wave Leap API key
# If USE_DWAVE_SAMPLER is False, a local Simulated Annealer will be used.
# Always use OR-Tools as a powerful classical baseline.
USE_OR_TOOLS_SOLVER = True