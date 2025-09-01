# environment.py (Full, Corrected, and Completed)

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Tuple
from opensimplex import OpenSimplex
from rtree import index

import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class Building:
    id: int
    center_xy: Tuple[float, float]
    size_xy: Tuple[float, float]
    height: float
    bounds: tuple = field(init=False)

    def __post_init__(self):
        # Calculate and store the 3D bounding box for efficient queries
        half_size_x, half_size_y = self.size_xy[0] / 2, self.size_xy[1] / 2
        min_x = self.center_xy[0] - half_size_x
        min_y = self.center_xy[1] - half_size_y
        max_x = self.center_xy[0] + half_size_x
        max_y = self.center_xy[1] + half_size_y
        # Store as (min_x, min_y, min_z, max_x, max_y, max_z)
        self.bounds = (min_x, min_y, 0, max_x, max_y, self.height)

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
        wind_x = wind_x_noise * self.max_speed
        wind_y = wind_y_noise * self.max_speed
        return np.array([wind_x, wind_y, 0])

class ObstacleIndex:
    def __init__(self):
        logging.info("Building spatial index for obstacles...")
        p = index.Property()
        p.dimension = 3
        self.idx = index.Index(properties=p)
        logging.info("Spatial index ready.")

    def insert(self, item_id, bounds):
        self.idx.insert(item_id, bounds)

    def intersection(self, bounds):
        return self.idx.intersection(bounds)

    def count(self, bounds):
        return len(list(self.idx.intersection(bounds)))

class Environment:
    def __init__(self, weather_system: WeatherSystem):
        self.static_nfzs = config.NO_FLY_ZONES
        self.buildings = self._generate_buildings()
        self.weather = weather_system
        self.obstacle_index = self._build_obstacle_index()
        self.dynamic_nfzs = []
        self.event_triggered = False

    def _generate_buildings(self):
        buildings = []
        np.random.seed(42)  # for reproducibility
        for i in range(70): # A reasonable number of buildings
            center_xy = (
                np.random.uniform(config.AREA_BOUNDS[0], config.AREA_BOUNDS[2]),
                np.random.uniform(config.AREA_BOUNDS[1], config.AREA_BOUNDS[3])
            )
            # Ensure buildings don't spawn inside static NFZs
            in_nfz = any(
                zone[0] < center_xy[0] < zone[2] and zone[1] < center_xy[1] < zone[3]
                for zone in self.static_nfzs
            )
            if in_nfz:
                continue
            
            size_xy = (np.random.uniform(0.0002, 0.0006), np.random.uniform(0.0002, 0.0006))
            height = np.random.uniform(50, 350)
            buildings.append(Building(id=i, center_xy=center_xy, size_xy=size_xy, height=height))
        return buildings

    def _build_obstacle_index(self):
        obstacle_idx = ObstacleIndex()
        # Add buildings
        for building in self.buildings:
            obstacle_idx.insert(building.id, building.bounds)
        
        # Add static No-Fly Zones
        # We give them IDs starting where the building IDs left off
        nfz_start_id = len(self.buildings)
        for i, zone in enumerate(self.static_nfzs):
            bounds = (zone[0], zone[1], 0, zone[2], zone[3], config.MAX_ALTITUDE)
            obstacle_idx.insert(nfz_start_id + i, bounds)
            
        logging.info(f"Spatial index populated with {len(self.buildings)} buildings and {len(self.static_nfzs)} NFZs.")
        return obstacle_idx

    def update_environment(self, simulation_time: float, time_step: float):
        """Evolves the environment over time."""
        self.weather.update_weather(time_step)
        
        # Example of a dynamic event: a temporary NFZ appears after 120s
        if simulation_time > 120 and not self.event_triggered:
            logging.info("EVENT: New No-Fly Zone activated!")
            new_nfz = [-74.005, 40.74, -73.995, 40.75]
            self.dynamic_nfzs.append(new_nfz)
            self.event_triggered = True

    def get_obstacle_by_id(self, obstacle_id: int):
        """
        Retrieves an obstacle's data by its unique ID.
        This is a required helper for the planner's efficient intersection tests.
        """
        # The ID from the R-tree is the index in the buildings list.
        if 0 <= obstacle_id < len(self.buildings):
            return self.buildings[obstacle_id]
        
        # Handle NFZs
        nfz_start_id = len(self.buildings)
        if nfz_start_id <= obstacle_id < nfz_start_id + len(self.static_nfzs):
            nfz_index = obstacle_id - nfz_start_id
            zone = self.static_nfzs[nfz_index]
            # Create a temporary object with bounds for the planner to use
            nfz_obj = type('NFZ', (object,), {
                'bounds': (zone[0], zone[1], 0, zone[2], zone[3], config.MAX_ALTITUDE)
            })()
            return nfz_obj

        return None