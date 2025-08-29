# environment.py
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import config

@dataclass
class Drone:
    id: int
    location: Tuple[float, float, float] = config.HUB_LOCATION
    
@dataclass
class Order:
    id: int
    location: Tuple[float, float, float]
    payload_kg: float
    
@dataclass
class Building:
    id: int
    center_xy: Tuple[float, float]
    size_xy: Tuple[float, float] # (length_x, width_y) in degrees
    height: float

class Environment:
    def __init__(self, num_drones, orders: List[Order]):
        self.drones: List[Drone] = [Drone(id=i) for i in range(num_drones)]
        self.orders: List[Order] = orders
        self.wind_vector: np.ndarray = np.array([2.0, -1.0, 0.0])
        self.buildings: List[Building] = self._generate_buildings()

    def _generate_buildings(self):
        """Generates random rectangular buildings."""
        buildings = []
        for i in range(50): # Increased number of buildings
            center_xy = (
                np.random.uniform(config.AREA_BOUNDS[0], config.AREA_BOUNDS[2]),
                np.random.uniform(config.AREA_BOUNDS[1], config.AREA_BOUNDS[3])
            )
            # 0.0001 degrees is ~11 meters. Buildings are ~22m to ~66m long/wide.
            size_xy = (np.random.uniform(0.0002, 0.0006), np.random.uniform(0.0002, 0.0006))
            height = np.random.uniform(50, 350)
            buildings.append(Building(id=i, center_xy=center_xy, size_xy=size_xy, height=height))
        return buildings