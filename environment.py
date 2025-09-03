import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict
from opensimplex import OpenSimplex
from rtree import index
import logging

from config import AREA_BOUNDS, MIN_ALTITUDE, MAX_ALTITUDE, NO_FLY_ZONES
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
        self.static_nfzs = NO_FLY_ZONES
        self.weather: WeatherSystem = weather_system
        self.dynamic_nfzs: List[Dict] = []
        self.event_triggered = False
        self.was_nfz_just_added = False

        p = index.Property()
        p.dimension = 3
        self.obstacle_index = index.Index(properties=p)
        self.obstacle_counter = 0
        self.obstacles = {}

        logging.info("Building spatial index for obstacles...")
        self.buildings: List[Building] = self._generate_and_index_buildings()
        self._index_static_nfzs()
        logging.info(f"Spatial index ready. Indexed {len(self.obstacles)} obstacles.")

    def _add_obstacle_to_index(self, bounds: tuple):
        obstacle_id = self.obstacle_counter
        self.obstacle_index.insert(obstacle_id, bounds)
        self.obstacles[obstacle_id] = bounds
        self.obstacle_counter += 1
        return obstacle_id

    def _generate_and_index_buildings(self) -> List[Building]:
        buildings = []
        np.random.seed(42)
        num_buildings = 20
        for i in range(num_buildings):
            center_x = np.random.uniform(AREA_BOUNDS[0], AREA_BOUNDS[2])
            center_y = np.random.uniform(AREA_BOUNDS[1], AREA_BOUNDS[3])
            width = np.random.uniform(0.001, 0.003)
            height_building = np.random.uniform(0.001, 0.003)
            altitude = np.random.uniform(50, MAX_ALTITUDE)
            
            building = Building(id=i, center_xy=(center_x, center_y), size_xy=(width, height_building), height=altitude)
            buildings.append(building)
            
            half_width, half_height = width / 2, height_building / 2
            bounds = (center_x - half_width, center_y - half_height, 0, center_x + half_width, center_y + half_height, altitude)
            self._add_obstacle_to_index(bounds)
        return buildings

    def _index_static_nfzs(self):
        for zone in self.static_nfzs:
            bounds = (zone[0], zone[1], MIN_ALTITUDE, zone[2], zone[3], MAX_ALTITUDE)
            self._add_obstacle_to_index(bounds)

    def remove_dynamic_obstacles(self):
        if not self.dynamic_nfzs:
            return
        logging.info(f"Clearing {len(self.dynamic_nfzs)} dynamic obstacles...")
        for d_nfz in self.dynamic_nfzs:
            obs_id = d_nfz['id']
            bounds = d_nfz['bounds']
            self.obstacle_index.delete(obs_id, bounds)
            if obs_id in self.obstacles:
                del self.obstacles[obs_id]
        
        self.dynamic_nfzs.clear()
        self.event_triggered = False
        self.was_nfz_just_added = False

    def is_point_obstructed(self, point: Tuple[float, float, float]) -> bool:
        """Check if a single point is inside any obstacle."""
        x, y, z = point
        if not (AREA_BOUNDS[0] <= x <= AREA_BOUNDS[2] and
                AREA_BOUNDS[1] <= y <= AREA_BOUNDS[3] and
                MIN_ALTITUDE <= z <= MAX_ALTITUDE):
            return True
        
        candidates = list(self.obstacle_index.intersection((x, y, z, x, y, z)))
        
        # The R-tree only provides potential candidates.
        # This loop confirms if the point is truly inside one of their bounding boxes.
        for obstacle_id in candidates:
            bounds = self.obstacles[obstacle_id]
            if (bounds[0] <= x <= bounds[3] and
                bounds[1] <= y <= bounds[4] and
                bounds[2] <= z <= bounds[5]):
                return True
        return False

    def is_line_obstructed(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> bool:
        min_x, min_y, min_z = min(p1[0], p2[0]), min(p1[1], p2[1]), min(p1[2], p2[2])
        max_x, max_y, max_z = max(p1[0], p2[0]), max(p1[1], p2[1]), max(p1[2], p2[2])
        
        candidate_ids = list(self.obstacle_index.intersection((min_x, min_y, min_z, max_x, max_y, max_z)))
        
        for obs_id in candidate_ids:
            bounds = self.obstacles[obs_id]
            if line_segment_intersects_aabb(p1, p2, bounds):
                return True
        return False

    def update_environment(self, simulation_time: float, time_step: float):
        self.weather.update_weather(time_step)
        
        if simulation_time > 15 and not self.event_triggered:
            logging.info("EVENT: New dynamic No-Fly Zone activated!")
            zone = [-74.005, 40.72, -73.995, 40.73] 
            bounds = (zone[0], zone[1], MIN_ALTITUDE, zone[2], zone[3], MAX_ALTITUDE)
            obs_id = self._add_obstacle_to_index(bounds)
            self.dynamic_nfzs.append({'zone': zone, 'bounds': bounds, 'id': obs_id})
            
            self.event_triggered = True
            self.was_nfz_just_added = True