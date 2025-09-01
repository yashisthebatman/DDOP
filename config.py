# config.py (Full Updated Code)
"""Central configuration file for the Q-DOP project."""

# --- Simulation Environment ---
AREA_BOUNDS = [-74.02, 40.70, -73.97, 40.74]
MIN_ALTITUDE = 50
MAX_ALTITUDE = 600
TAKEOFF_ALTITUDE = 50

# --- Strategic Planning Constant ---
DEFAULT_CRUISING_ALTITUDE = 400 # meters

# --- Hub Locations & Destinations (Unchanged) ---
HUBS = {"Hub A (South Manhattan)":(-74.013,40.705,10),"Hub B (Midtown East)":(-73.975,40.740,10),"Hub C (West Side)":(-74.005,40.735,10)}
DESTINATIONS = {"One World Trade":(-74.0134,40.7127,400.0),"Empire State Building":(-73.9857,40.7484,381.0),"NYU Campus":(-73.9962,40.7295,50.0),"Hudson Yards Vessel":(-74.0025,40.7538,50.0),"South Street Seaport":(-74.0036,40.706,50.0),"Wall Street Bull":(-74.0134,40.7056,50.0),"Madison Square Garden":(-73.9936,40.7505,70.0),"StuyTown Apartments":(-73.9780,40.7320,80.0),"Chelsea Market":(-74.0060,40.7423,50.0),"Union Square":(-73.9904,40.7359,50.0)}

# --- No-Fly Zones (Unchanged) ---
NO_FLY_ZONES = [[-74.01,40.715,-73.995,40.725],[-73.985,40.735,-73.975,40.745]]

# --- Drone & Physics (Unchanged) ---
DRONE_SPEED_MPS=25;DRONE_MAX_PAYLOAD_KG=5.0;DRONE_BATTERY_WH=20.0;DRONE_MASS_KG=2.0;ASCENT_EFFICIENCY=0.7;GRAVITY=9.81;TURN_ENERGY_FACTOR=0.005

# --- Pathfinding ---
# The "greediness" of the algorithm. 1.0 is optimal. > 1.0 is faster but less optimal.
A_STAR_HEURISTIC_WEIGHT = 1.2
# --- NEW: D* Lite Planner Settings ---
# If True, runs a one-time, potentially slow pre-computation of wind effects
# across the entire map. If False, uses a default cost of 1.0 for open air.
PRECOMPUTE_WIND_COSTS = True
# Default cost multiplier for traversing an open-air grid cell.
DEFAULT_CELL_COST = 1.0