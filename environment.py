# environment.py
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from shapely.geometry import Polygon, Point, LineString
import config

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

class Environment:
    """Manages the state of the physical environment for the simulation."""
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.bounds = config.AREA_BOUNDS
        self.drones: List[Drone] = []
        self.orders: List[Order] = []
        self.charging_pads: List[ChargingPad] = []
        self.no_fly_zones: List[Polygon] = []
        self.buildings: List[Building] = []
        self.wind_vector: np.ndarray = self._generate_wind()
        self._create_scenario()

    def _generate_wind(self):
        return np.array([5.0, 0.0, 0.0])

    def _create_scenario(self):
        """Generates a random but repeatable scenario."""
        for i in range(config.NUM_DRONES):
            loc = (0, np.random.uniform(self.bounds[1], self.bounds[3]), 100)
            self.drones.append(Drone(id=i, start_location=loc, current_location=loc))

        for i in range(config.NUM_ORDERS):
            loc = (
                np.random.uniform(self.bounds[0], self.bounds[2]),
                np.random.uniform(self.bounds[1], self.bounds[3]),
                np.random.uniform(50, 150)
            )
            payload = np.random.uniform(0.5, config.DRONE_MAX_PAYLOAD_KG - 1)
            self.orders.append(Order(id=i, location=loc, payload_kg=payload))
        
        for i in range(config.NUM_BUILDINGS):
            center_xy = (
                np.random.uniform(self.bounds[0] * 0.2, self.bounds[2] * 0.8),
                np.random.uniform(self.bounds[1] * 0.2, self.bounds[3] * 0.8)
            )
            radius = np.random.uniform(100, 300)
            height = np.random.uniform(100, 250)
            self.buildings.append(Building(id=i, center_xy=center_xy, radius=radius, height=height))

        center_x = (self.bounds[0] + self.bounds[2]) / 2
        center_y = (self.bounds[1] + self.bounds[3]) / 2
        nfz = Polygon([(center_x-500, center_y-500), (center_x+500, center_y-500), 
                       (center_x+500, center_y+500), (center_x-500, center_y+500)])
        self.no_fly_zones.append(nfz)

    def is_path_valid(self, p1, p2):
        """Checks if a straight line path violates any 2D no-fly zones."""
        line = LineString([Point(p1[0], p1[1]), Point(p2[0], p2[1])])
        for zone in self.no_fly_zones:
            if line.intersects(zone):
                return False
        return True