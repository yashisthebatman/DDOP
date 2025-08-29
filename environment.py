# environment.py
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
# CHANGE: Added LineString to the import list
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

class Environment:
    """Manages the state of the physical environment for the simulation."""
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.bounds = config.AREA_BOUNDS
        self.drones: List[Drone] = []
        self.orders: List[Order] = []
        self.charging_pads: List[ChargingPad] = []
        self.no_fly_zones: List[Polygon] = []
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
            
        for i in range(config.NUM_CHARGING_PADS):
             loc = (
                np.random.uniform(self.bounds[0], self.bounds[2]),
                np.random.uniform(self.bounds[1], self.bounds[3]),
                0
            )
             self.charging_pads.append(ChargingPad(id=i, location=loc))

        center_x = (self.bounds[0] + self.bounds[2]) / 2
        center_y = (self.bounds[1] + self.bounds[3]) / 2
        nfz = Polygon([(center_x-500, center_y-500), (center_x+500, center_y-500), 
                       (center_x+500, center_y+500), (center_x-500, center_y+500)])
        self.no_fly_zones.append(nfz)

    def is_path_valid(self, p1, p2):
        """Checks if a straight line path violates any no-fly zones."""
        # For simplicity, we check in 2D (top-down view)
        line = LineString([Point(p1[0], p1[1]), Point(p2[0], p2[1])])
        for zone in self.no_fly_zones:
            if line.intersects(zone):
                return False
        return True

    def get_all_locations(self) -> List[Tuple]:
        locations = [d.start_location for d in self.drones]
        locations += [o.location for o in self.orders]
        locations += [p.location for p in self.charging_pads]
        return locations

    def get_location_map(self):
        loc_map = {}
        idx = 0
        for drone in self.drones:
            loc_map[idx] = {'type': 'depot', 'obj': drone}
            idx += 1
        for order in self.orders:
            loc_map[idx] = {'type': 'order', 'obj': order}
            idx += 1
        for pad in self.charging_pads:
            loc_map[idx] = {'type': 'pad', 'obj': pad}
            idx += 1
        return loc_map