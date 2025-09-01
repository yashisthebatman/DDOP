# config.py (Full, Corrected, and Completed)
"""Central configuration file for the Q-DOP project."""

# --- Simulation Environment ---
AREA_BOUNDS = [-74.02, 40.70, -73.97, 40.74] # [lon_min, lat_min, lon_max, lat_max]
MIN_ALTITUDE = 10.0 # meters
MAX_ALTITUDE = 200.0 # meters
TAKEOFF_ALTITUDE = 15.0 # meters
GRID_RESOLUTION = 0.0001 # in degrees, approx 11 meters

# --- Strategic Planning Constant ---
DEFAULT_CRUISING_ALTITUDE = 100.0 # meters

# --- Hub Locations & Destinations ---
HUBS = {
    "Hub A (South Manhattan)": (-74.013, 40.705, 10.0),
    "Hub B (Midtown East)": (-73.975, 40.740, 10.0),
    "Hub C (West Side)": (-74.005, 40.735, 10.0)
}
DESTINATIONS = {
    "One World Trade": (-74.0134, 40.7127, 150.0),
    "Empire State Building": (-73.9857, 40.7484, 150.0),
    "NYU Campus": (-73.9962, 40.7295, 50.0),
    "Hudson Yards Vessel": (-74.0025, 40.7538, 50.0),
    "South Street Seaport": (-74.0036, 40.706, 50.0),
    "Wall Street Bull": (-74.0134, 40.7056, 50.0),
    "Madison Square Garden": (-73.9936, 40.7505, 70.0),
    "StuyTown Apartments": (-73.9780, 40.7320, 80.0),
    "Chelsea Market": (-74.0060, 40.7423, 50.0),
    "Union Square": (-73.9904, 40.7359, 50.0)
}

# --- No-Fly Zones (Static Obstacles) ---
NO_FLY_ZONES = [
    [-74.01, 40.715, -73.995, 40.725],
    [-73.985, 40.735, -73.975, 40.745]
]

# --- Drone & Physics ---
DRONE_SPEED_MPS = 20.0
DRONE_MAX_PAYLOAD_KG = 5.0
DRONE_BATTERY_WH = 200.0
DRONE_MASS_KG = 2.0
ASCENT_EFFICIENCY = 0.7
GRAVITY = 9.81
TURN_ENERGY_FACTOR = 0.005 # Energy cost per degree of turn

# --- Pathfinding ---
# The "greediness" of the algorithm. 1.0 is optimal. > 1.0 is faster but less optimal.
A_STAR_HEURISTIC_WEIGHT = 1.2
# --- NEW: Safety limit for JPS recursion to prevent stack overflow ---
MAX_PATH_LENGTH = 5000 # A safe upper limit on path grid steps