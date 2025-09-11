"""Central configuration file for the Q-DOP project."""
import warnings

# --- Simulation Environment ---

AREA_BOUNDS = [-74.02, 40.70, -73.97, 40.74]  # [lon_min, lat_min, lon_max, lat_max]
MIN_ALTITUDE = 10.0  # meters
MAX_ALTITUDE = 200.0  # meters
TAKEOFF_ALTITUDE = 15.0  # meters

# --- Tactical Grid Configuration for A* and D* Lite ---

GRID_RESOLUTION_M = 15 # Horizontal resolution (in meters) for the grid
GRID_VERTICAL_RESOLUTION_M = 5 # Vertical resolution (in meters) for the grid

# --- RRT* Strategic Planner (for single-agent RTH) ---

# FIX: The original RRT* parameters were not well-suited for a dense,
# obstacle-filled environment. A smaller step size is more likely to find
# a path through the gaps between buildings.
RRT_ITERATIONS = 5000       # Number of nodes to try (increased for more exploration)
RRT_STEP_SIZE_METERS = 75.0  # How far to extend the tree in one step (decreased for finer navigation)
RRT_GOAL_BIAS = 0.1         # Probability of sampling the goal point (0.0 to 1.0)
RRT_NEIGHBORHOOD_RADIUS_METERS = 100.0 # Radius to search for rewiring (decreased, but still > step_size)

# --- Hub Locations & Destinations ---

HUBS = {
    "Hub A (South Manhattan)": (-74.013, 40.705, 10.0),
    "Hub B (Midtown East)": (-73.975, 40.738, 10.0),
    "Hub C (West Side)": (-74.005, 40.735, 10.0)
}
DESTINATIONS = {
    "One World Trade": (-74.0134, 40.7127, 100.0),
    "Empire State Building": (-73.9857, 40.739, 150.0),
    "NYU Campus": (-73.9962, 40.7295, 50.0),
    "Hudson Yards Vessel": (-74.0025, 40.739, 50.0),
    "South Street Seaport": (-74.0036, 40.706, 50.0),
    "Wall Street Bull": (-74.0134, 40.7056, 50.0),
    "Madison Square Garden": (-73.9936, 40.7395, 70.0),
    "StuyTown Apartments": (-73.9780, 40.7320, 80.0),
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
DRONE_BASE_POWER_WATTS = 50.0
DRONE_ADDITIONAL_WATTS_PER_KG = 10.0
RTH_BATTERY_THRESHOLD_FACTOR = 1.5 # Trigger RTH if energy to hub * this factor > remaining battery

# --- Pathfinding Parameters ---
MAX_PATH_LENGTH = 5000

# --- Configuration Validation ---

def validate_coordinates():
    lon_min, lat_min, lon_max, lat_max = AREA_BOUNDS
    all_points = {**HUBS, **DESTINATIONS}
    errors = []
    for name, (lon, lat, alt) in all_points.items():
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            errors.append(f"Invalid coordinate for '{name}': ({lon}, {lat}) is outside AREA_BOUNDS.")
    if errors:
        for error in errors:
            warnings.warn(f"Configuration error: {error}")

validate_coordinates()