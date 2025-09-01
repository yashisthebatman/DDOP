import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from opensimplex import OpenSimplex
from rtree import index
import config
import logging
from utils.geometry import line_segment_intersects_aabb

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
        if seed is None:
            seed = np.random.randint(0, 1000)
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

        # R-tree for fast spatial queries
        p = index.Property()
        p.dimension = 3
        self.obstacle_index = index.Index(properties=p)
        self.obstacle_counter = 0
        
        # Store obstacle bounds for precise checks
        self.obstacles = {}

        logging.info("Building spatial index for obstacles...")
        self.buildings: List[Building] = self._generate_and_index_buildings()
        self._index_static_nfzs()
        logging.info(f"Spatial index ready. Indexed {self.obstacle_counter} obstacles.")

    def _add_obstacle_to_index(self, bounds: tuple):
        """Adds a 3D bounding box to the R-tree and local storage."""
        obstacle_id = self.obstacle_counter
        self.obstacle_index.insert(obstacle_id, bounds)
        self.obstacles[obstacle_id] = bounds
        self.obstacle_counter += 1
        return obstacle_id

    def _generate_and_index_buildings(self) -> List[Building]:
        """Generate buildings and add them to the spatial index."""
        buildings = []
        np.random.seed(42)
        
        num_buildings = 20
        for i in range(num_buildings):
            center_x = np.random.uniform(config.AREA_BOUNDS[0], config.AREA_BOUNDS[2])
            center_y = np.random.uniform(config.AREA_BOUNDS[1], config.AREA_BOUNDS[3])
            
            width = np.random.uniform(0.001, 0.003)
            height_building = np.random.uniform(0.001, 0.003)
            altitude = np.random.uniform(50, 300)
            
            building = Building(
                id=i,
                center_xy=(center_x, center_y),
                size_xy=(width, height_building),
                height=altitude
            )
            buildings.append(building)
            
            half_width = width / 2
            half_height = height_building / 2
            bounds = (
                center_x - half_width, center_y - half_height, 0,
                center_x + half_width, center_y + half_height, altitude
            )
            self._add_obstacle_to_index(bounds)
        
        return buildings

    def _index_static_nfzs(self):
        """Add static no-fly zones to the spatial index."""
        for zone in self.static_nfzs:
            bounds = (zone[0], zone[1], 0, zone[2], zone[3], config.MAX_ALTITUDE)
            self._add_obstacle_to_index(bounds)

    def get_all_nfzs(self) -> List[List[float]]:
        """Get all no-fly zones (static + dynamic)."""
        return self.static_nfzs + self.dynamic_nfzs

    def get_obstacle_by_id(self, obs_id):
        """Get obstacle bounds by ID."""
        if obs_id in self.obstacles:
            bounds = self.obstacles[obs_id]
            class Obstacle:
                def __init__(self, bounds):
                    self.bounds = bounds
            return Obstacle(bounds)
        return None

    def is_point_obstructed(self, point: Tuple[float, float, float]) -> bool:
        """Check if a single point is obstructed by any obstacle."""
        x, y, z = point
        
        # Check bounds first
        if not (config.AREA_BOUNDS[0] <= x <= config.AREA_BOUNDS[2] and
                config.AREA_BOUNDS[1] <= y <= config.AREA_BOUNDS[3] and
                config.MIN_ALTITUDE <= z <= config.MAX_ALTITUDE):
            return True
        
        # Use R-tree for fast spatial query
        candidates = list(self.obstacle_index.intersection((x, y, z, x, y, z)))
        
        for obstacle_id in candidates:
            bounds = self.obstacles[obstacle_id]
            if (bounds[0] <= x <= bounds[3] and
                bounds[1] <= y <= bounds[4] and
                bounds[2] <= z <= bounds[5]):
                return True
        
        return False

    def is_line_obstructed(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> bool:
        """Check if a line segment between two points intersects any obstacle."""
        min_x, min_y, min_z = min(p1[0], p2[0]), min(p1[1], p2[1]), min(p1[2], p2[2])
        max_x, max_y, max_z = max(p1[0], p2[0]), max(p1[1], p2[1]), max(p1[2], p2[2])
        
        candidates = list(self.obstacle_index.intersection((min_x, min_y, min_z, max_x, max_y, max_z)))
        
        for obstacle_id in candidates:
            bounds = self.obstacles[obstacle_id]
            if line_segment_intersects_aabb(p1, p2, bounds):
                return True
        
        return False

    def update_environment(self, simulation_time: float, time_step: float):
        """Update the environment state (weather, dynamic obstacles)."""
        self.weather.update_weather(time_step)
        
        if simulation_time > 120 and not self.event_triggered:
            logging.info("EVENT: New No-Fly Zone activated!")
            zone = [-74.005, 40.74, -73.995, 40.75]
            self.dynamic_nfzs.append(zone)
            bounds = (zone[0], zone[1], 0, zone[2], zone[3], config.MAX_ALTITUDE)
            self._add_obstacle_to_index(bounds)
            self.event_triggered = True
            self.was_nfz_just_added = True