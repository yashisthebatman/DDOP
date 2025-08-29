# environment.py
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from shapely.geometry import Polygon
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
    size_xy: Tuple[float, float]
    height: float

@dataclass
class WeatherZone:
    """Defines a polygonal area with a specific wind vector."""
    id: int
    polygon: Polygon
    wind_vector: np.ndarray # (x, y, z) m/s

class WeatherSystem:
    """Manages different weather zones within the simulation area."""
    def __init__(self, zones: List[WeatherZone]):
        self.zones = zones
        self.default_wind = np.array([0.0, 0.0, 0.0]) # Default calm wind

    def get_wind_at_location(self, lon, lat):
        """Gets the wind vector for a specific world coordinate."""
        from shapely.geometry import Point
        point = Point(lon, lat)
        for zone in self.zones:
            if zone.polygon.contains(point):
                return zone.wind_vector
        return self.default_wind

class Environment:
    def __init__(self, num_drones, orders: List[Order], weather_system: WeatherSystem):
        self.drones: List[Drone] = [Drone(id=i) for i in range(num_drones)]
        self.orders: List[Order] = orders
        self.weather: WeatherSystem = weather_system
        self.buildings: List[Building] = self._generate_buildings()

    def _generate_buildings(self):
        """Generates random rectangular buildings."""
        buildings = []
        np.random.seed(42) # for reproducibility
        for i in range(75):
            center_xy = (
                np.random.uniform(config.AREA_BOUNDS[0], config.AREA_BOUNDS[2]),
                np.random.uniform(config.AREA_BOUNDS[1], config.AREA_BOUNDS[3])
            )
            size_xy = (np.random.uniform(0.0002, 0.0006), np.random.uniform(0.0002, 0.0006))
            height = np.random.uniform(50, 350)
            buildings.append(Building(id=i, center_xy=center_xy, size_xy=size_xy, height=height))
        return buildings