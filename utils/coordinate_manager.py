# FILE: utils/coordinate_manager.py
import logging
from typing import Tuple, Optional
import numpy as np

from config import (
    AREA_BOUNDS, MIN_ALTITUDE, MAX_ALTITUDE,
    GRID_RESOLUTION_M, GRID_VERTICAL_RESOLUTION_M
)

class CoordinateManager:
    """
    Manages conversions between World (lon/lat), Local Meters, and Grid coordinates.
    """
    def __init__(self):
        self.lon_min, self.lat_min, self.lon_max, self.lat_max = AREA_BOUNDS
        self.alt_min, self.alt_max = MIN_ALTITUDE, MAX_ALTITUDE
        
        # FIX: Store the grid resolution as an attribute
        self.grid_res_m = GRID_RESOLUTION_M
        
        ref_lat_rad = np.radians(self.lat_min + (self.lat_max - self.lat_min) / 2)
        self.lon_deg_to_m = 111320 * np.cos(ref_lat_rad)
        self.lat_deg_to_m = 110574

        self.origin_world = (self.lon_min, self.lat_min, 0)

        self.area_width_m = (self.lon_max - self.lon_min) * self.lon_deg_to_m
        self.area_height_m = (self.lat_max - self.lat_min) * self.lat_deg_to_m
        self.grid_width = int(self.area_width_m / self.grid_res_m)
        self.grid_height = int(self.area_height_m / self.grid_res_m)
        self.grid_depth = int((self.alt_max - self.alt_min) / GRID_VERTICAL_RESOLUTION_M)


    def world_to_meters(self, world_pos: Tuple) -> Tuple[float, float, float]:
        """Converts World (lon, lat, alt) to Local Meters (x, y, z)."""
        lon, lat, alt = world_pos
        x = (lon - self.lon_min) * self.lon_deg_to_m
        y = (lat - self.lat_min) * self.lat_deg_to_m
        return (x, y, alt)

    def meters_to_world(self, meters_pos: Tuple) -> Tuple[float, float, float]:
        """Converts Local Meters (x, y, z) back to World (lon, lat, alt)."""
        mx, my, alt = meters_pos
        lon = (mx / self.lon_deg_to_m) + self.lon_min
        lat = (my / self.lat_deg_to_m) + self.lat_min
        return lon, lat, alt

    def meters_to_grid(self, meters_pos: Tuple) -> Optional[Tuple[int, int, int]]:
        """Converts Local Meters to the discretized Grid coordinate system."""
        mx, my, alt = meters_pos
        grid_x = int(mx / self.grid_res_m)
        grid_y = int(my / self.grid_res_m)
        grid_z = int((alt - self.alt_min) / GRID_VERTICAL_RESOLUTION_M)
        
        grid_pos = (grid_x, grid_y, grid_z)
        return grid_pos if self.is_valid_grid_pos(grid_pos) else None
        
    def grid_to_meters(self, grid_pos: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Converts a Grid coordinate to its corresponding center point in Local Meters."""
        gx, gy, gz = grid_pos
        mx = (gx + 0.5) * self.grid_res_m
        my = (gy + 0.5) * self.grid_res_m
        alt = (gz + 0.5) * GRID_VERTICAL_RESOLUTION_M + self.alt_min
        return (mx, my, alt)

    def is_valid_grid_pos(self, grid_pos: Tuple[int, int, int]) -> bool:
        """Checks if a grid coordinate is within the defined area bounds."""
        x, y, z = grid_pos
        return (0 <= x < self.grid_width and
                0 <= y < self.grid_height and
                0 <= z < self.grid_depth)