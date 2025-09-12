# FILE: environment.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict
from opensimplex import OpenSimplex
from rtree import index
import logging

from config import AREA_BOUNDS, MIN_ALTITUDE, MAX_ALTITUDE, NO_FLY_ZONES
from utils.geometry import line_segment_intersects_aabb
from utils.coordinate_manager import CoordinateManager

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

class RiskMap:
    def __init__(self):
        self.risk_grid = np.random.rand(10, 10) * 0.1
        self.risk_grid[3:6, 3:6] = 0.9
    def get_risk(self, world_pos: Tuple[float, float, float]) -> float:
        lon_idx = int(np.interp(world_pos[0], [AREA_BOUNDS[0], AREA_BOUNDS[2]], [0, 9]))
        lat_idx = int(np.interp(world_pos[1], [AREA_BOUNDS[1], AREA_BOUNDS[3]], [0, 9]))
        return self.risk_grid[np.clip(lat_idx, 0, 9), np.clip(lon_idx, 0, 9)]

class WeatherSystem:
    def __init__(self, seed=None, scale=150.0, max_speed=15.0):
        if seed is None: seed = np.random.randint(0, 1000)
        self.noise_gen_x = OpenSimplex(seed)
        self.noise_gen_y = OpenSimplex(seed + 1)
        self.scale = scale
        self.max_speed = max_speed
        self.time = 0
    def update_weather(self, time_step=0.01): self.time += time_step
    def get_wind_at_location(self, lon: float, lat: float, alt: float) -> np.ndarray:
        norm_lon, norm_lat = lon / self.scale, lat / self.scale
        wind_x_noise = self.noise_gen_x.noise3(norm_lon, norm_lat, self.time)
        wind_y_noise = self.noise_gen_y.noise3(norm_lon, norm_lat, self.time)
        altitude_factor = np.interp(alt, [MIN_ALTITUDE, MAX_ALTITUDE], [0.7, 1.2])
        effective_max_speed = self.max_speed * altitude_factor
        wind_x = wind_x_noise * effective_max_speed
        wind_y = wind_y_noise * effective_max_speed
        return np.array([wind_x, wind_y, 0])

class Environment:
    def __init__(self, weather_system: WeatherSystem, coord_manager: CoordinateManager):
        self.coord_manager = coord_manager
        self.static_nfzs = NO_FLY_ZONES
        self.weather: WeatherSystem = weather_system
        self.risk_map: RiskMap = RiskMap()
        self.dynamic_nfzs: List[Dict] = []
        self.event_triggered = False
        self.was_nfz_just_added = False
        p = index.Property()
        p.dimension = 3
        self.obstacle_index = index.Index(properties=p)
        self.obstacle_counter = 0
        self.obstacles = {}
        logging.info("Building spatial index for obstacles in meter-space...")
        self.buildings: List[Building] = self._generate_and_index_buildings()
        self._index_static_nfzs()
        logging.info(f"Spatial index ready. Indexed {len(self.obstacles)} obstacles.")

    def _add_obstacle_to_index(self, bounds_m: tuple):
        obstacle_id = self.obstacle_counter
        self.obstacle_index.insert(obstacle_id, bounds_m)
        self.obstacles[obstacle_id] = bounds_m
        self.obstacle_counter += 1
        return obstacle_id

    def _generate_and_index_buildings(self) -> List[Building]:
        buildings = []
        np.random.seed(42)
        for i in range(20):
            center_x = np.random.uniform(AREA_BOUNDS[0], AREA_BOUNDS[2])
            center_y = np.random.uniform(AREA_BOUNDS[1], AREA_BOUNDS[3])
            width_deg, height_deg = np.random.uniform(0.0001, 0.0003), np.random.uniform(0.0001, 0.0003)
            altitude = np.random.uniform(50, MAX_ALTITUDE)
            building = Building(id=i, center_xy=(center_x, center_y), size_xy=(width_deg, height_deg), height=altitude)
            buildings.append(building)
            bottom_left_world = (center_x - width_deg / 2, center_y - height_deg / 2, 0)
            top_right_world = (center_x + width_deg / 2, center_y + height_deg / 2, altitude)
            min_mx, min_my, _ = self.coord_manager.world_to_meters(bottom_left_world)
            max_mx, max_my, _ = self.coord_manager.world_to_meters(top_right_world)
            self._add_obstacle_to_index((min_mx, min_my, 0, max_mx, max_my, altitude))
        return buildings

    def _index_static_nfzs(self):
        for zone in self.static_nfzs:
            bottom_left_world = (zone[0], zone[1], MIN_ALTITUDE)
            top_right_world = (zone[2], zone[3], MAX_ALTITUDE)
            min_mx, min_my, _ = self.coord_manager.world_to_meters(bottom_left_world)
            max_mx, max_my, _ = self.coord_manager.world_to_meters(top_right_world)
            self._add_obstacle_to_index((min_mx, min_my, MIN_ALTITUDE, max_mx, max_my, MAX_ALTITUDE))
            
    def remove_dynamic_obstacles(self):
        if not self.dynamic_nfzs: return
        for d_nfz in self.dynamic_nfzs:
            self.obstacle_index.delete(d_nfz['id'], d_nfz['bounds'])
            if d_nfz['id'] in self.obstacles: del self.obstacles[d_nfz['id']]
        self.dynamic_nfzs.clear()
        self.event_triggered = self.was_nfz_just_added = False
    
    def add_dynamic_nfz(self, zone_world: List[float]):
        """Adds a new dynamic No-Fly Zone to the environment."""
        bl_world, tr_world = (zone_world[0], zone_world[1], MIN_ALTITUDE), (zone_world[2], zone_world[3], MAX_ALTITUDE)
        min_mx, min_my, _ = self.coord_manager.world_to_meters(bl_world)
        max_mx, max_my, _ = self.coord_manager.world_to_meters(tr_world)
        bounds_m = (min_mx, min_my, MIN_ALTITUDE, max_mx, max_my, MAX_ALTITUDE)
        obs_id = self._add_obstacle_to_index(bounds_m)
        self.dynamic_nfzs.append({'zone': zone_world, 'bounds': bounds_m, 'id': obs_id})
        self.event_triggered = self.was_nfz_just_added = True
        logging.info(f"Added new dynamic NFZ with ID {obs_id} at bounds {zone_world}")

    def is_point_obstructed(self, point_world: Tuple[float, float, float]) -> bool:
        lon, lat, alt = point_world
        if not (AREA_BOUNDS[0] <= lon <= AREA_BOUNDS[2] and AREA_BOUNDS[1] <= lat <= AREA_BOUNDS[3] and MIN_ALTITUDE <= alt <= MAX_ALTITUDE):
            return True
        point_m = self.coord_manager.world_to_meters(point_world)
        mx, my, mz = point_m
        for obs_id in self.obstacle_index.intersection((mx, my, mz, mx, my, mz)):
            bounds_m = self.obstacles[obs_id]
            if (bounds_m[0] <= mx <= bounds_m[3] and bounds_m[1] <= my <= bounds_m[4] and bounds_m[2] <= mz <= bounds_m[5]):
                return True
        return False

    def is_line_obstructed(self, p1_world: Tuple[float, float, float], p2_world: Tuple[float, float, float]) -> bool:
        p1_m, p2_m = np.array(self.coord_manager.world_to_meters(p1_world)), np.array(self.coord_manager.world_to_meters(p2_world))
        min_m, max_m = np.minimum(p1_m, p2_m), np.maximum(p1_m, p2_m)
        for obs_id in self.obstacle_index.intersection(tuple(np.concatenate((min_m, max_m)))):
            if line_segment_intersects_aabb(tuple(p1_m), tuple(p2_m), self.obstacles[obs_id]):
                return True
        return False
    
    def update_environment(self, simulation_time: float, time_step: float):
        self.weather.update_weather(time_step)
        # The logic for adding a dynamic NFZ is now handled externally
        # by the event injector to allow for random, unpredictable events.
        
    def create_planning_grid(self) -> np.ndarray:
        w, h, d = self.coord_manager.grid_width, self.coord_manager.grid_height, self.coord_manager.grid_depth
        grid = np.full((w, h, d), True)
        logging.info(f"Creating planning grid of size ({w}, {h}, {d})...")
        for obs_id, bounds_m in self.obstacles.items():
            min_mx, min_my, min_mz, max_mx, max_my, max_mz = bounds_m
            min_g, max_g = self.coord_manager.meters_to_grid((min_mx, min_my, min_mz)), self.coord_manager.meters_to_grid((max_mx, max_my, max_mz))
            if min_g and max_g:
                min_gx, min_gy, min_gz = min_g
                max_gx, max_gy, max_gz = max_g
                grid[max(0, min_gx):min(w, max_gx + 1), max(0, min_gy):min(h, max_gy + 1), max(0, min_gz):min(d, max_gz + 1)] = False
        logging.info("Planning grid created.")
        return grid