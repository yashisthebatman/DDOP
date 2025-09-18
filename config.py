# FILE: config.py
"""Central configuration file for the Q-DOP project."""
import warnings

# --- Simulation Environment ---
AREA_BOUNDS = [-74.02, 40.70, -73.97, 40.74]
MIN_ALTITUDE = 10.0
MAX_ALTITUDE = 200.0
TAKEOFF_ALTITUDE = 15.0

# --- App Simulation Control ---
SIMULATION_TIME_STEP = 0.5
SIMULATION_UI_REFRESH_INTERVAL = 0.05
MIN_ORDERS_TO_DISPATCH = 1
MAX_CONCURRENT_PLANNERS = 4
DRONES_PER_HUB = 5
# NEW: Add a safety margin to pre-flight energy checks to account for unknown pathing/wind.
DISPATCH_ENERGY_SAFETY_MARGIN = 1.2 # (20% buffer)

# --- ML Model and Retraining Configuration ---
MODEL_FILE_PATH = "ml_predictor/drone_predictor_model.joblib"
TRAINING_DATA_PATH = "training/training_data.csv"
RETRAINING_THRESHOLD = 20

# --- Tactical Grid Configuration ---
GRID_RESOLUTION_M = 15
GRID_VERTICAL_RESOLUTION_M = 5
COARSE_GRID_RESOLUTION_M = 50

# --- RRT* Strategic Planner ---
RRT_STEP_SIZE_METERS = 50.0
RRT_GOAL_BIAS = 0.05
RRT_NEIGHBORHOOD_RADIUS_METERS = 100.0

# --- Hub Locations & Destinations ---
HUBS = [
    {"id": "HUB_A", "name": "Hub A (South Manhattan)", "location": (-74.018, 40.705, 10.0)},
    {"id": "HUB_B", "name": "Hub B (Midtown East)", "location": (-73.975, 40.729, 10.0)},
    {"id": "HUB_C", "name": "Hub C (West Side)", "location": (-74.008, 40.735, 10.0)}
]
DESTINATIONS = {
    "One World Trade": (-74.0134, 40.7127, 100.0),
    "Empire State Building": (-73.9860, 40.739, 150.0),
    "NYU Campus": (-73.9962, 40.7295, 50.0),
    "Hudson Yards Vessel": (-74.0025, 40.739, 50.0),
    "South Street Seaport": (-74.0036, 40.706, 50.0),
    "Wall Street Bull": (-74.0134, 40.7056, 50.0),
    "Madison Square Garden": (-73.9936, 40.7395, 70.0),
    "StuyTown Apartments": (-73.9740, 40.7320, 80.0),
    "Chelsea Market": (-74.0060, 40.738, 50.0),
    "Union Square": (-73.9904, 40.7359, 50.0)
}

# --- No-Fly Zones (Static Obstacles) ---
NO_FLY_ZONES = [
    [-74.01, 40.715, -73.995, 40.725],
    [-73.985, 40.73, -73.975, 40.74]
]

# --- Drone & Physics ---
DRONE_SPEED_MPS = 20.0
DRONE_VERTICAL_SPEED_MPS = 5.0
DRONE_MAX_PAYLOAD_KG = 5.0
DRONE_BATTERY_WH = 200.0
DRONE_MASS_KG = 2.0
ASCENT_EFFICIENCY = 0.7
GRAVITY = 9.81
TURN_ENERGY_FACTOR = 0.005
# NOTE: These power values are also used for calculating hover/delivery maneuver energy drain.
DRONE_BASE_POWER_WATTS = 50.0
DRONE_ADDITIONAL_WATTS_PER_KG = 10.0
RTH_BATTERY_THRESHOLD_FACTOR = 1.5
DRONE_RECHARGE_TIME_S = 30.0

# --- Pathfinding Parameters ---
MAX_PATH_LENGTH = 5000
PATH_SMOOTHING_OBSTACLE_CLEARANCE_METERS = 5.0

# --- Delivery Maneuver ---
DELIVERY_MANEUVER_TIME_SEC = 90

# --- Deconfliction Parameters ---
SAFETY_BUBBLE_RADIUS_METERS = 30.0
AVOIDANCE_MANEUVER_ALTITUDE_SEP = 15.0