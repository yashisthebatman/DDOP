import numpy as np
from dataclasses import dataclass
from typing import Tuple
from opensimplex import OpenSimplex
from rtree import index
import config
import logging

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
        self.weather: WeatherSystem = weather_system
        self.dynamic_nfzs = []
        self.event_triggered = False
        self.was_nfz_just_added = False # Used by app to trigger replan

        # --- NEW: R-tree for fast spatial queries ---
        p = index.Property()
        p.dimension = 3
        self.obstacle_index = index.Index(properties=p)
        self.obstacle_counter = 0
        
        logging.info("Building spatial index for obstacles...")
        self.buildings: list[Building] = self._generate_and_index_buildings()
        self._index_static_nfzs()
        logging.info(f"Spatial index ready. Indexed {self.obstacle_counter} obstacles.")


    def _add_obstacle_to_index(self, bounds: tuple, is_dynamic=False):
        """Adds a 3D bounding box to the R-tree index."""
        # bounds = (min_lon, min_lat, min_alt, max_lon, max_lat, max_alt)
        self.obstacle_index.insert(self.obstacle_counter, bounds)
        self.obstacle_counter += 1

    def update_environment(self, simulation_time: float, time_step: float):
        """Evolves the environment over time."""
        self.weather.update_weather(time_step)
        
        # Example of a dynamic event: a temporary NFZ appears after 120s
        if simulation_time > 120 and not self.event_triggered:
            logging.info("EVENT: New No-Fly Zone activated!")
            zone = [-74.005, 40.74, -73.995, 40.75] # Near Chelsea/Hudson Yards
            self.dynamic_nfzs.append(zone)
            bounds = (zone[0], zone[1], 0, zone[2], zone[3], config.MAX_ALTITUDE)
            self._add_obstacle_to_index(bounds, is_dynamic=True)
            self.event_triggered = True
            self.was_nfz_just_added = True


    def is_point_obstructed(self, point: Tuple[float, float, float]) -> bool:
        """Checks if a single 3D world coordinate is inside any obstacle."""
        # A point is a zero-volume box
        return self.obstacle_index.count(point * 2) > 0

    def is_line_obstructed(self, p1: tuple, p2: tuple, samples=20) -> bool:
        """Checks if a line segment between two points intersects any obstacle."""
        # Sample points along the line and check each one
        points_to_check = np.linspace(p1, p2, samples)
        for p in points_to_check:
            if self.is_point_obstructed(tuple(p)):
                return True
        return False

    def get_all_nfzs(self) -> list:
        """Returns a combined list of static and dynamic NFZs."""
        return self.static_nfzs + self.dynamic_nfzs

    def _generate_and_index_buildings(self):
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
            
            # Add to R-tree index
            min_lon, max_lon = center_xy[0] - size_xy[0]/2, center_xy[0] + size_xy[0]/2
            min_lat, max_lat = center_xy[1] - size_xy[1]/2, center_xy[1] + size_xy[1]/2
            bounds = (min_lon, min_lat, 0, max_lon, max_lat, height)
            self._add_obstacle_to_index(bounds)

            buildings.append(Building(id=i, center_xy=center_xy, size_xy=size_xy, height=height))
        return buildings
    
    def _index_static_nfzs(self):
        """Adds all static No-Fly Zones to the R-tree index."""
        for zone in self.static_nfzs:
            bounds = (zone[0], zone[1], 0, zone[2], zone[3], config.MAX_ALTITUDE)
            self._add_obstacle_to_index(bounds)