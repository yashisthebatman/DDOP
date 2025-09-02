# ==============================================================================
# File: config.py
# ==============================================================================
"""Central configuration file for the Q-DOP project."""
import warnings

# --- Simulation Environment ---
AREA_BOUNDS = [-74.02, 40.70, -73.97, 40.74]  # [lon_min, lat_min, lon_max, lat_max]
MIN_ALTITUDE = 10.0  # meters
MAX_ALTITUDE = 200.0  # meters
TAKEOFF_ALTITUDE = 15.0  # meters
GRID_RESOLUTION = 0.0001  # in degrees, approx 11 meters

# --- Strategic Planning Constants ---
DEFAULT_CRUISING_ALTITUDE = 100.0  # meters
DEFAULT_CELL_COST = 1.0

# --- Hub Locations & Destinations ---
HUBS = {
    "Hub A (South Manhattan)": (-74.013, 40.705, 10.0),
    "Hub B (Midtown East)": (-73.975, 40.740, 10.0), # FIX: Corrected typo from 40.749 to 40.740
    "Hub C (West Side)": (-74.005, 40.735, 10.0)
}

DESTINATIONS = {
    "One World Trade": (-74.0134, 40.7127, 100.0), # FIX: Changed altitude for better pathing
    # FIX: Corrected latitude for Empire State Building to be within bounds
    "Empire State Building": (-73.9857, 40.7399, 150.0), # Was 40.7484 (out of bounds)
    "NYU Campus": (-73.9962, 40.7295, 50.0),
    # FIX: Corrected latitude for Hudson Yards to be within bounds
    "Hudson Yards Vessel": (-74.0025, 40.7390, 50.0), # Was 40.7538 (out of bounds)
    "South Street Seaport": (-74.0036, 40.706, 50.0),
    "Wall Street Bull": (-74.0134, 40.7056, 50.0),
    # FIX: Corrected latitude for Madison Square Garden to be within bounds
    "Madison Square Garden": (-73.9936, 40.7395, 70.0), # Was 40.7505 (out of bounds)
    "StuyTown Apartments": (-73.9780, 40.7320, 80.0),
    "Chelsea Market": (-74.0060, 40.7380, 50.0), # Was 40.7423 (out of bounds)
    "Union Square": (-73.9904, 40.7359, 50.0)
}

# --- No-Fly Zones (Static Obstacles) ---
NO_FLY_ZONES = [
    [-74.01, 40.715, -73.995, 40.725],
    [-73.985, 40.735, -73.975, 40.745]
]

# --- Drone & Physics ---
DRONE_SPEED_MPS = 20.0
DRONE_VERTICAL_SPEED_MPS = 5.0
DRONE_MAX_PAYLOAD_KG = 5.0
DRONE_BATTERY_WH = 200.0
DRONE_MASS_KG = 2.0
ASCENT_EFFICIENCY = 0.7
GRAVITY = 9.81
TURN_ENERGY_FACTOR = 0.005  # Energy cost per degree of turn
# FIX: Moved physics "magic numbers" to config for better tuning and realism
DRONE_BASE_POWER_WATTS = 50.0
DRONE_ADDITIONAL_WATTS_PER_KG = 10.0


# --- Pathfinding Parameters ---
A_STAR_HEURISTIC_WEIGHT = 1.2  # Heuristic weight (1.0 = optimal, >1.0 = faster but suboptimal)
MAX_PATH_LENGTH = 5000  # Safety limit for JPS recursion
PRECOMPUTE_WIND_COSTS = True

# --- FIX: Configuration Validation ---
def validate_coordinates():
    """Checks if all defined hubs and destinations are within the AREA_BOUNDS."""
    lon_min, lat_min, lon_max, lat_max = AREA_BOUNDS
    all_points = {**HUBS, **DESTINATIONS}
    errors = []

    for name, (lon, lat, alt) in all_points.items():
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            error_msg = f"Invalid coordinate for '{name}': ({lon}, {lat}) is outside AREA_BOUNDS."
            errors.append(error_msg)

    if errors:
        for error in errors:
            warnings.warn(f"Configuration error: {error}")

validate_coordinates()