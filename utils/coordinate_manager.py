# ==============================================================================
# File: utils/coordinate_manager.py
# ==============================================================================
import logging
from typing import Tuple, Optional

import numpy as np

from config import (
    AREA_BOUNDS,
    MIN_ALTITUDE,
    MAX_ALTITUDE,
    GRID_RESOLUTION_M,
    GRID_VERTICAL_RESOLUTION_M
)

class CoordinateManager:
    def __init__(self):
        self.lon_min, self.lat_min, self.lon_max, self.lat_max = AREA_BOUNDS
        self.alt_min, self.alt_max = MIN_ALTITUDE, MAX_ALTITUDE
        self.grid_size_m = 3000
        self.grid_width = int(self.grid_size_m / GRID_RESOLUTION_M)
        self.grid_height = int(self.grid_size_m / GRID_RESOLUTION_M)
        self.grid_depth = int((self.alt_max - self.alt_min) / GRID_VERTICAL_RESOLUTION_M)
        self.local_grid_origin_world = (self.lon_min, self.lat_min, self.alt_min)
        self.ref_lat = np.radians(self.lat_min + (self.lat_max - self.lat_min) / 2)
        self.lon_deg_to_m = 111320 * np.cos(self.ref_lat)
        self.lat_deg_to_m = 110574

    def set_local_grid_origin(self, world_pos: Tuple[float, float, float]):
        self.local_grid_origin_world = world_pos

    def world_to_local_meters(self, world_pos: Tuple) -> Tuple[float, float, float]:
        lon, lat, alt = world_pos
        return (lon - self.lon_min) * self.lon_deg_to_m, (lat - self.lat_min) * self.lat_deg_to_m, alt

    def world_to_local_grid(self, world_pos: Tuple) -> Optional[Tuple[int, int, int]]:
        origin_m = self.world_to_local_meters(self.local_grid_origin_world)
        target_m = self.world_to_local_meters(world_pos)
        grid_pos = (int(round((target_m[0] - origin_m[0]) / GRID_RESOLUTION_M)),
                    int(round((target_m[1] - origin_m[1]) / GRID_RESOLUTION_M)),
                    int((world_pos[2] - self.alt_min) / GRID_VERTICAL_RESOLUTION_M))
        return grid_pos if self.is_valid_local_grid_pos(grid_pos) else None

    def local_grid_to_world(self, grid_pos: Optional[Tuple] = None, base_world_pos: Optional[Tuple] = None, offset_m: Optional[Tuple] = None) -> Optional[Tuple[float, float, float]]:
        # LOG: Log the function call arguments to see exactly what we're getting
        logging.debug(f"local_grid_to_world called with: grid_pos={grid_pos}, base_world_pos={base_world_pos}, offset_m={offset_m}")

        # FIX: Restructured the function to prevent the logic from falling through and returning None erroneously.
        # Each use case now has an explicit and immediate return.
        
        # Use Case 1: Apply a meter-based offset (for RRT* steering)
        if base_world_pos is not None and offset_m is not None:
            base_lon, base_lat, base_alt = base_world_pos
            dx_m, dy_m, dz_m = offset_m
            new_lon = base_lon + (dx_m / self.lon_deg_to_m)
            new_lat = base_lat + (dy_m / self.lat_deg_to_m)
            new_alt = base_alt + dz_m
            logging.debug(f"RRT* steering case calculated result: {(new_lon, new_lat, new_alt)}")
            return new_lon, new_lat, new_alt
        
        # Use Case 2: Convert a D* Lite grid coordinate
        if grid_pos is not None:
            gx, gy, gz = grid_pos
            origin_lon, origin_lat, _ = self.local_grid_origin_world
            new_lon = origin_lon + ((gx * GRID_RESOLUTION_M) / self.lon_deg_to_m)
            new_lat = origin_lat + ((gy * GRID_RESOLUTION_M) / self.lat_deg_to_m)
            new_alt = self.alt_min + (gz * GRID_VERTICAL_RESOLUTION_M)
            return new_lon, new_lat, new_alt
        
        # This log is critical. If we see it, it means the function was called with bad arguments.
        logging.error("local_grid_to_world called with invalid arguments, returning None.")
        return None

    def is_valid_local_grid_pos(self, pos: Tuple) -> bool:
        x, y, z = pos
        return (-self.grid_width//2 <= x < self.grid_width//2 and
                -self.grid_height//2 <= y < self.grid_height//2 and
                0 <= z < self.grid_depth)