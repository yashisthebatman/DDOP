# environment.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple
import config

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

class WeatherSystem:
    """Manages a single, uniform wind vector for the entire simulation area."""
    def __init__(self, wind_speed_mps, wind_direction_deg):
        angle_rad = np.deg2rad(270 - wind_direction_deg)
        self.wind_vector = np.array([
            np.cos(angle_rad) * wind_speed_mps,
            np.sin(angle_rad) * wind_speed_mps,
            0
        ])

    def get_wind_at_location(self, lon, lat):
        """Gets the global wind vector."""
        return self.wind_vector

class Environment:
    def __init__(self, wind_speed_mps, wind_direction_deg):
        # BUG FIX: Initialize no_fly_zones *before* calling _generate_buildings
        self.no_fly_zones = config.NO_FLY_ZONES
        self.buildings: list[Building] = self._generate_buildings()
        self.weather: WeatherSystem = WeatherSystem(wind_speed_mps, wind_direction_deg)

    def _generate_buildings(self):
        """Generates random rectangular buildings."""
        buildings = []
        np.random.seed(42) # for reproducibility
        for i in range(75):
            center_xy = (
                np.random.uniform(config.AREA_BOUNDS[0], config.AREA_BOUNDS[2]),
                np.random.uniform(config.AREA_BOUNDS[1], config.AREA_BOUNDS[3])
            )
            is_in_nfz = False
            for zone in self.no_fly_zones: # This line now works correctly
                if zone[0] < center_xy[0] < zone[2] and zone[1] < center_xy[1] < zone[3]:
                    is_in_nfz = True
                    break
            if is_in_nfz: continue
            
            size_xy = (np.random.uniform(0.0002, 0.0006), np.random.uniform(0.0002, 0.0006))
            height = np.random.uniform(50, 350)
            buildings.append(Building(id=i, center_xy=center_xy, size_xy=size_xy, height=height))
        return buildings