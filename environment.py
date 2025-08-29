# environment.py
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from shapely.geometry import Polygon, Point, LineString
import config

# --- THIS IS THE CORRECTED DATACLASS SECTION ---
@dataclass
class Drone:
    id: int
    start_location: Tuple[float, float, float]
    current_location: Tuple[float, float, float]
    max_payload_kg: float = config.DRONE_MAX_PAYLOAD_KG
    battery_wh: float = config.DRONE_MAX_BATTERY_WH
    current_payload_kg: float = 0.0

@dataclass
class Order:
    id: int
    location: Tuple[float, float, float]
    payload_kg: float

@dataclass
class ChargingPad:
    id: int
    location: Tuple[float, float, float]

@dataclass
class Building:
    id: int
    center_xy: Tuple[float, float]
    radius: float
    height: float
# --- END OF CORRECTION ---

class Environment:
    """Manages the state of the physical environment for the simulation."""
    def __init__(self, num_drones, num_orders, num_buildings, seed=42):
        np.random.seed(seed)
        self.bounds = config.AREA_BOUNDS
        self.drones: List[Drone] = []
        self.orders: List[Order] = []
        self.buildings: List[Building] = []
        self.wind_vector: np.ndarray = self._generate_wind()
        self._create_scenario(num_drones, num_orders, num_buildings)

    def _generate_wind(self):
        return np.array([5.0, 0.0, 0.0])

    def _create_scenario(self, num_drones, num_orders, num_buildings):
        """Generates a scenario based on user-defined parameters."""
        # Create Drones
        for i in range(num_drones):
            loc = (0, np.random.uniform(self.bounds[1], self.bounds[3]), config.MIN_ALTITUDE)
            self.drones.append(Drone(id=i, start_location=loc, current_location=loc))

        # Create Orders
        for i in range(num_orders):
            loc = (
                np.random.uniform(self.bounds[0], self.bounds[2]),
                np.random.uniform(self.bounds[1], self.bounds[3]),
                np.random.uniform(config.MIN_ALTITUDE, 250) # Orders are below max building height
            )
            payload = np.random.uniform(0.5, config.DRONE_MAX_PAYLOAD_KG - 1)
            self.orders.append(Order(id=i, location=loc, payload_kg=payload))
        
        # Create Buildings
        for i in range(num_buildings):
            center_xy = (
                np.random.uniform(self.bounds[0] * 0.2, self.bounds[2] * 0.8),
                np.random.uniform(self.bounds[1] * 0.2, self.bounds[3] * 0.8)
            )
            radius = np.random.uniform(100, 300)
            height = np.random.uniform(150, 450) # Varied building heights
            self.buildings.append(Building(id=i, center_xy=center_xy, radius=radius, height=height))

    def is_path_valid(self, p1, p2):
        # This function is now less critical as A* handles 3D obstacles,
        # but can be kept for broad 2D no-fly zones if needed.
        return True