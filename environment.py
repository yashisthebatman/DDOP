import numpy as np
from dataclasses import dataclass
from typing import Tuple
import config
from opensimplex import OpenSimplex # Import the new library

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
    """
    Manages a dynamic, non-uniform wind field using simplex noise.
    The wind pattern can evolve over time.
    """
    def __init__(self, seed=None, scale=150.0, max_speed=15.0):
        """
        Initializes the dynamic weather system.
        :param seed: A seed for the noise generator for reproducibility.
        :param scale: Controls the "size" of wind gusts. Larger scale = broader wind patterns.
        :param max_speed: The maximum possible wind speed in m/s.
        """
        if seed is None:
            seed = np.random.randint(0, 1000)
        self.noise_gen_x = OpenSimplex(seed)
        self.noise_gen_y = OpenSimplex(seed + 1) # Use a different seed for the y-component
        self.scale = scale
        self.max_speed = max_speed
        self.time = 0 # Time component for evolving weather

    def update_weather(self, time_step=0.1):
        """Call this to make the wind pattern evolve over time."""
        self.time += time_step

    def get_wind_at_location(self, lon, lat):
        """
        Gets the specific wind vector for any world coordinate.
        The wind now varies smoothly across the map and over time.
        """
        # Normalize coordinates to be used with the noise function
        norm_lon, norm_lat = lon / self.scale, lat / self.scale
        # Get noise values (between -1 and 1) incorporating the time component
        wind_x_noise = self.noise_gen_x.noise3(norm_lon, norm_lat, self.time)
        wind_y_noise = self.noise_gen_y.noise3(norm_lon, norm_lat, self.time)
        # Map noise values to the desired wind speed range
        wind_x, wind_y = wind_x_noise * self.max_speed, wind_y_noise * self.max_speed
        return np.array([wind_x, wind_y, 0])

class Environment:
    def __init__(self, weather_system: WeatherSystem):
        # BUG FIX: Initialize no_fly_zones *before* calling _generate_buildings
        self.no_fly_zones = config.NO_FLY_ZONES
        self.buildings: list[Building] = self._generate_buildings()
        self.weather: WeatherSystem = weather_system

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
            for zone in self.no_fly_zones:
                if zone[0] < center_xy[0] < zone[2] and zone[1] < center_xy[1] < zone[3]:
                    is_in_nfz = True; break
            if is_in_nfz: continue
            
            size_xy = (np.random.uniform(0.0002, 0.0006), np.random.uniform(0.0002, 0.0006))
            height = np.random.uniform(50, 350)
            buildings.append(Building(id=i, center_xy=center_xy, size_xy=size_xy, height=height))
        return buildings