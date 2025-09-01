import numpy as np
from dataclasses import dataclass
from typing import Tuple
from opensimplex import OpenSimplex
from rtree import index
import config
import logging
from utils.geometry import line_segment_intersects_aabb # <-- Import the new function

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
        self.was_nfz_just_added = False

        # --- R-tree for fast spatial queries ---
        p = index.Property(); p.dimension = 3
        self.obstacle_index = index.Index(properties=p)
        self.obstacle_counter = 0
        
        # --- NEW: Store obstacle bounds for precise checks ---
        self.obstacles = {}

        logging.info("Building spatial index for obstacles...")
        self.buildings: list[Building] = self._generate_and_index_buildings()
        self._index_static_nfzs()
        logging.info(f"Spatial index ready. Indexed {self.obstacle_counter} obstacles.")


    def _add_obstacle_to_index(self, bounds: tuple):
        """Adds a 3D bounding box to the R-tree and local storage."""
        obstacle_id = self.obstacle_counter
        self.obstacle_index.insert(obstacle_id, bounds)
        self.obstacles[obstacle_id] = bounds # Store bounds for precise checks
        self.obstacle_counter += 1

    def update_environment(self, simulation_time: float, time_step: float):
        self.weather.update_weather(time_step)
        if simulation_time > 120 and not self.event_triggered:
            logging.info("EVENT: New No-Fly Zone activated!")
            zone = [-74.005, 40.74, -73.995, 40.75]
            self.dynamic_nfzs.append(zone)
            bounds = (zone[0], zone[1], 0, zone[2], zone[3], config.MAX_ALTITUDE)
            self._add_obstacle_to_index(bounds)
            self.event_triggered = True
            self.was_nfz_just_added = True

    def is_point_obstructed(self, point: Tuple[float, float, float]) -> bool:
        """Checks if a single 3D world coordinate is inside any obstacle."""
        return self.obstacle_index.count(point * 2) > 0

    def is_line_obstructed(self, p1: tuple, p2: tuple) -> bool:
        """
        OPTIMIZED: Checks if a line segment intersects any obstacle in the R-tree.
        First performs a broad-phase check, then a precise narrow-phase check.
        """
        # 1. Broad Phase: Get a bounding box for the entire path segment.
        path_bounds = (
            min(p1[0], p2[0]), min(p1[1], p2[1]), min(p1[2], p2[2]),
            max(p1[0], p2[0]), max(p1[1], p2[1]), max(p1[2], p2[2])
        )

        # 2. Query the R-tree for any obstacles that intersect this path's bounding box.
        potential_colliders_ids = list(self.obstacle_index.intersection(path_bounds))
        
        # If the broad-phase check finds nothing, the path is clear.
        if not potential_colliders_ids:
            return False
            
        # 3. Narrow Phase: For the few potential colliders, perform a precise
        #    line segment vs. AABB intersection test.
        for obstacle_id in potential_colliders_ids:
            obstacle_bounds = self.obstacles[obstacle_id]
            if line_segment_intersects_aabb(p1, p2, obstacle_bounds):
                return True # Found a definite collision
        
        # If none of the potential colliders actually intersected the line segment, the path is clear.
        return False


    def get_all_nfzs(self) -> list:
        return self.static_nfzs + self.dynamic_nfzs

    def _generate_and_index_buildings(self):
        buildings = []
        np.random.seed(42)
        for i in range(75):
            center_xy = (
                np.random.uniform(config.AREA_BOUNDS[0], config.AREA_BOUNDS[2]),
                np.random.uniform(config.AREA_BOUNDS[1], config.AREA_BOUNDS[3])
            )
            if any(z[0]<center_xy[0]<z[2] and z[1]<center_xy[1]<z[3] for z in self.static_nfzs):
                continue
            
            size_xy = (np.random.uniform(0.0002, 0.0006), np.random.uniform(0.0002, 0.0006))
            height = np.random.uniform(50, 350)
            
            min_lon, max_lon = center_xy[0] - size_xy[0]/2, center_xy[0] + size_xy[0]/2
            min_lat, max_lat = center_xy[1] - size_xy[1]/2, center_xy[1] + size_xy[1]/2
            bounds = (min_lon, min_lat, 0, max_lon, max_lat, height)
            self._add_obstacle_to_index(bounds)

            buildings.append(Building(id=i, center_xy=center_xy, size_xy=size_xy, height=height))
        return buildings
    
    def _index_static_nfzs(self):
        for zone in self.static_nfzs:
            bounds = (zone[0], zone[1], 0, zone[2], zone[3], config.MAX_ALTITUDE)
            self._add_obstacle_to_index(bounds)