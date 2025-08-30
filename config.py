# config.py
"""Central configuration file for the Q-DOP project."""

# --- (Environment, Hubs, Destinations, NFZs, Drone Physics are unchanged) ---
AREA_BOUNDS = [-74.02, 40.70, -73.97, 40.74]
MIN_ALTITUDE = 50
MAX_ALTITUDE = 600
TAKEOFF_ALTITUDE = 50
HUBS = {"Hub A (South Manhattan)": (-74.013, 40.705, 10), "Hub B (Midtown East)": (-73.975, 40.740, 10), "Hub C (West Side)": (-74.005, 40.735, 10)}
DESTINATIONS = {"One World Trade": (-74.0134, 40.7127, 400.0), "Empire State Building": (-73.9857, 40.7484, 381.0), "NYU Campus": (-73.9962, 40.7295, 50.0), "Hudson Yards Vessel": (-74.0025, 40.7538, 50.0), "South Street Seaport": (-74.0036, 40.706, 50.0), "Wall Street Bull": (-74.0134, 40.7056, 50.0), "Madison Square Garden": (-73.9936, 40.7505, 70.0), "StuyTown Apartments": (-73.9780, 40.7320, 80.0), "Chelsea Market": (-74.0060, 40.7423, 50.0), "Union Square": (-73.9904, 40.7359, 50.0), "Google Building": (-74.0030, 40.7400, 150.0), "New Museum": (-73.9928, 40.7223, 60.0), "Pier 40": (-74.0118, 40.7288, 15.0), "City Hall": (-74.0064, 40.7128, 55.0), "Battery Park": (-74.0170, 40.7033, 50.0)}
NO_FLY_ZONES = [[-74.01, 40.715, -73.995, 40.725], [-73.985, 40.735, -73.975, 40.745]]
DRONE_SPEED_MPS = 25
DRONE_MAX_PAYLOAD_KG = 5.0
DRONE_BATTERY_WH = 20.0
DRONE_MASS_KG = 2.0
ACCELERATION_ENERGY_BASE_WH = (0.5 * 1.0 * DRONE_SPEED_MPS**2) / 3600
TURN_ENERGY_FACTOR = 0.005

# --- QUBO Solver Strategy ---
QUBO_PRIMARY_SWEEPS = 1000   # Standard number of sweeps for the annealer
QUBO_RECOVERY_SWEEPS = 2500  # Increased sweeps for difficult problems