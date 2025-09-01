# environment.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from opensimplex import OpenSimplex
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
    def __init__(self, seed=None, scale=150.0, max_speed=15.0):
        if seed is None: seed = np.random.randint(0, 1000)
        self.noise_gen_x = OpenSimplex(seed)
        self.noise_gen_y = OpenSimplex(seed + 1)
        self.scale = scale
        self.max_speed = max_speed
        self.time = 0

    def update_weather(self, time_step=0.01):
        self.time += time_step

    def get_wind_at_location(self, lon, lat):
        norm_lon, norm_lat = lon / self.scale, lat / self.scale
        wind_x_noise = self.noise_gen_x.noise3(norm_lon, norm_lat, self.time)
        wind_y_noise = self.noise_gen_y.noise3(norm_lon, norm_lat, self.time)
        wind_x, wind_y = wind_x_noise * self.max_speed, wind_y_noise * self.max_speed
        return np.array([wind_x, wind_y, 0])

class Environment:
    def __init__(self, weather_system: WeatherSystem):
        self.static_nfzs = config.NO_FLY_ZONES
        self.buildings: list[Building] = self._generate_buildings()
        self.weather: WeatherSystem = weather_system
        # --- New dynamic elements ---
        self.dynamic_nfzs = []
        self.event_triggered = False

    def update_environment(self, simulation_time: float, time_step: float):
        """Evolves the environment over time."""
        self.weather.update_weather(time_step)
        
        # Example of a dynamic event: a temporary NFZ appears after 120s
        if simulation_time > 120 and not self.event_triggered:
            print("EVENT: New No-Fly Zone activated!")
            self.dynamic_nfzs.append([-74.005, 40.74, -73.995, 40.75]) # Near Chelsea/Hudson Yards
            self.event_triggered = True

    def get_all_nfzs(self) -> list:
        """Returns a combined list of static and dynamic NFZs."""
        return self.static_nfzs + self.dynamic_nfzs

    def _generate_buildings(self):
        buildings = []
        np.random.seed(42)
        for i in range(75):
            center_xy = (
                np.random.uniform(config.AREA_BOUNDS[0], config.AREA_BOUNDS[2]),
                np.random.uniform(config.AREA_BOUNDS[1], config.AREA_BOUNDS[3])
            )
            is_in_nfz = any(
                zone[0] < center_xy[0] < zone[2] and zone[1] < center_xy[1] < zone[3]
                for zone in self.static_nfzs
            )
            if is_in_nfz: continue
            size_xy = (np.random.uniform(0.0002, 0.0006), np.random.uniform(0.0002, 0.0006))
            height = np.random.uniform(50, 350)
            buildings.append(Building(id=i, center_xy=center_xy, size_xy=size_xy, height=height))
        return buildings