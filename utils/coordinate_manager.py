# FILE: utils/coordinate_manager.py
import logging
from typing import Tuple, Optional
import numpy as np

from config import (
    AREA_BOUNDS, MIN_ALTITUDE, MAX_ALTITUDE,
    GRID_RESOLUTION_M, GRID_VERTICAL_RESOLUTION_M
)

class CoordinateManager:
    def __init__(self):
        self.lon_min, self.lat_min, self.lon_max, self.lat_max = AREA_BOUNDS
        self.alt_min, self.alt_max = MIN_ALTITUDE, MAX_ALTITUDE
        
        self.ref_lat = np.radians(self.lat_min + (self.lat_max - self.lat_min) / 2)
        self.lon_deg_to_m = 111320 * np.cos(self.ref_lat)
        self.lat_deg_to_m = 110574
        
        # Default grid setup, can be temporarily overridden
        self.set_local_grid_origin(
            world_pos=(self.lon_min, self.lat_min, self.alt_min),
            grid_size_m=max((self.lon_max - self.lon_min) * self.lon_deg_to_m, (self.lat_max - self.lat_min) * self.lat_deg_to_m)
        )

    def set_local_grid_origin(self, world_pos: Tuple[float, float, float], grid_size_m: Optional[float] = None):
        """Sets the center and size of the tactical grid."""
        self.local_grid_origin_world = world_pos
        if grid_size_m:
            self.grid_size_m = grid_size_m
        self.grid_width = int(self.grid_size_m / GRID_RESOLUTION_M)
        self.grid_height = int(self.grid_size_m / GRID_RESOLUTION_M)
        self.grid_depth = int((self.alt_max - self.alt_min) / GRID_VERTICAL_RESOLUTION_M)

    def world_to_local_meters(self, world_pos: Tuple) -> Tuple[float, float, float]:
        lon, lat, alt = world_pos
        return (lon - self.lon_min) * self.lon_deg_to_m, (lat - self.lat_min) * self.lat_deg_to_m, alt

    def local_meters_to_world(self, local_pos_m: Tuple) -> Tuple[float, float, float]:
        """Converts a local meter-based position back to world lon/lat/alt."""
        mx, my, alt = local_pos_m
        lon = (mx / self.lon_deg_to_m) + self.lon_min
        lat = (my / self.lat_deg_to_m) + self.lat_min
        return lon, lat, alt

    def world_to_local_grid(self, world_pos: Tuple) -> Optional[Tuple[int, int, int]]:
        origin_m = self.world_to_local_meters(self.local_grid_origin_world)
        target_m = self.world_to_local_meters(world_pos)
        
        grid_pos = (
            int(round((target_m[0] - origin_m[0]) / GRID_RESOLUTION_M)),
            int(round((target_m[1] - origin_m[1]) / GRID_RESOLUTION_M)),
            int((world_pos[2] - self.alt_min) / GRID_VERTICAL_RESOLUTION_M)
        )
        return grid_pos if self.is_valid_local_grid_pos(grid_pos) else None

    def local_grid_to_world(self, grid_pos: Tuple[int, int, int]) -> Optional[Tuple[float, float, float]]:
        gx, gy, gz = grid_pos
        origin_lon, origin_lat, _ = self.local_grid_origin_world
        
        new_lon = origin_lon + ((gx * GRID_RESOLUTION_M) / self.lon_deg_to_m)
        new_lat = origin_lat + ((gy * GRID_RESOLUTION_M) / self.lat_deg_to_m)
        new_alt = self.alt_min + (gz * GRID_VERTICAL_RESOLUTION_M)
        
        return new_lon, new_lat, new_alt
        
    def is_valid_local_grid_pos(self, pos: Tuple[int, int, int]) -> bool:
        x, y, z = pos
        half_width, half_height = self.grid_width // 2, self.grid_height // 2
        return (-half_width <= x < half_width and
                -half_height <= y < half_height and
                0 <= z < self.grid_depth)