# config.py
"""Central configuration file for the Q-DOP project."""

# --- Simulation Environment ---
AREA_BOUNDS = [-74.02, 40.70, -73.97, 40.74] # Bounding box for Lower Manhattan
MIN_ALTITUDE = 10
MAX_ALTITUDE = 600
TAKEOFF_ALTITUDE = 50

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
    "Hudson Yards Vessel": (-74.0025, 40.7538, 45.0),
    "South Street Seaport": (-74.0036, 40.706, 30.0),
    "Wall Street Bull": (-74.0134, 40.7056, 20.0),
    "Madison Square Garden": (-73.9936, 40.7505, 70.0),
    "StuyTown Apartments": (-73.9780, 40.7320, 80.0),
    "Chelsea Market": (-74.0060, 40.7423, 40.0),
    "Union Square": (-73.9904, 40.7359, 25.0),
    "Google Building": (-74.0030, 40.7400, 150.0),
    "New Museum": (-73.9928, 40.7223, 60.0),
    "Pier 40": (-74.0118, 40.7288, 15.0),
    "City Hall": (-74.0064, 40.7128, 55.0),
    "Battery Park": (-74.0170, 40.7033, 20.0)
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

# --- Advanced Physics Constants ---
# Energy (Wh) to accelerate 1kg to max speed. E = 0.5 * m * v^2 -> joules / 3600 -> Wh
ACCELERATION_ENERGY_BASE_WH = (0.5 * 1.0 * DRONE_SPEED_MPS**2) / 3600
# Energy cost per radian of turning. Higher values make the drone prefer straighter paths.
TURN_ENERGY_FACTOR = 0.005