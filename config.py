# config.py
"""Central configuration file for the Q-DOP project."""

# --- Simulation Environment ---
AREA_BOUNDS = [-74.02, 40.70, -73.97, 40.74]
MIN_ALTITUDE = 50
MAX_ALTITUDE = 600
TAKEOFF_ALTITUDE = 50

# --- Strategic Planning Constant ---
DEFAULT_CRUISING_ALTITUDE = 400 # meters

# --- Hub Locations ---
HUBS = {
    "Hub A (South Manhattan)": (-74.013, 40.705, 10),
    "Hub B (Midtown East)": (-73.975, 40.740, 10),
    "Hub C (West Side)": (-74.005, 40.735, 10)
}

# --- Delivery Destinations ---
DESTINATIONS = {
    "One World Trade": (-74.0134, 40.7127, 400.0),
    "Empire State Building": (-73.9857, 40.7484, 381.0),
    "NYU Campus": (-73.9962, 40.7295, 50.0),
    "Hudson Yards Vessel": (-74.0025, 40.7538, 50.0),
    "South Street Seaport": (-74.0036, 40.706, 50.0),
    "Wall Street Bull": (-74.0134, 40.7056, 50.0),
    "Madison Square Garden": (-73.9936, 40.7505, 70.0),
    "StuyTown Apartments": (-73.9780, 40.7320, 80.0),
    "Chelsea Market": (-74.0060, 40.7423, 50.0),
    "Union Square": (-73.9904, 40.7359, 50.0)
}

# --- No-Fly Zones [min_lon, min_lat, max_lon, max_lat] ---
NO_FLY_ZONES = [
    [-74.01, 40.715, -73.995, 40.725],
    [-73.985, 40.735, -73.975, 40.745]
]

# --- Drone & Physics ---
DRONE_SPEED_MPS = 25
DRONE_MAX_PAYLOAD_KG = 5.0
DRONE_BATTERY_WH = 20.0
DRONE_MASS_KG = 2.0
ASCENT_EFFICIENCY = 0.7 # How much more energy it takes to go up vs. glide down
GRAVITY = 9.81

# --- Advanced Physics Constants ---
TURN_ENERGY_FACTOR = 0.005

# --- Pathfinding ---
# The "greediness" of the A* algorithm. 1.0 is optimal (standard A*).
# > 1.0 is faster but may not find the absolute best path. 1.2-1.5 is a good balance.
A_STAR_HEURISTIC_WEIGHT = 1.2