import numpy as np
import logging
from typing import Tuple, Optional

# Import config variables directly
from config import AREA_BOUNDS, MIN_ALTITUDE, MAX_ALTITUDE, GRID_RESOLUTION_M, GRID_VERTICAL_RESOLUTION_M

class CoordinateManager:
    """
    Manages conversions between coordinate systems.
    1. Global World (lon, lat, alt) to Global Meters (for consistent distance calcs).
    2. World to a DYNAMIC Local Grid (for tactical planning like D* Lite).
    """
    def __init__(self):
        self.lon_min, self.lat_min, self.lon_max, self.lat_max = AREA_BOUNDS
        self.alt_min, self.alt_max = MIN_ALTITUDE, MAX_ALTITUDE
        
        self.grid_size_m = 3000
        self.grid_width = int(self.grid_size_m / GRID_RESOLUTION_M)
        self.grid_height = int(self.grid_size_m / GRID_RESOLUTION_M)
        self.grid_depth = int((self.alt_max - self.alt_min) / GRID_VERTICAL_RESOLUTION_M)
        self.grid_shape = (self.grid_width, self.grid_height, self.grid_depth)
        self.local_grid_origin_world = (self.lon_min, self.lat_min, self.alt_min)
        self.ref_lat = np.radians(self.lat_min + (self.lat_max - self.lat_min) / 2)
        self.lon_deg_to_m = 111320 * np.cos(self.ref_lat)
        self.lat_deg_to_m = 110574

    def set_local_grid_origin(self, world_pos: Tuple[float, float, float]):
        self.local_grid_origin_world = world_pos
        logging.debug(f"Set local grid origin to {world_pos}")

    def world_to_local_meters(self, world_pos: Tuple) -> Tuple[float, float, float]:
        lon, lat, alt = world_pos
        x_m = (lon - self.lon_min) * self.lon_deg_to_m
        y_m = (lat - self.lat_min) * self.lat_deg_to_m
        return x_m, y_m, alt

    def world_to_local_grid(self, world_pos: Tuple) -> Optional[Tuple[int, int, int]]:
        origin_m = self.world_to_local_meters(self.local_grid_origin_world)
        target_m = self.world_to_local_meters(world_pos)
        dx_m = target_m[0] - origin_m[0]
        dy_m = target_m[1] - origin_m[1]
        gx = int(round(dx_m / GRID_RESOLUTION_M))
        gy = int(round(dy_m / GRID_RESOLUTION_M))
        gz = int((world_pos[2] - self.alt_min) / GRID_VERTICAL_RESOLUTION_M)
        grid_pos = (gx, gy, gz)
        return grid_pos if self.is_valid_local_grid_pos(grid_pos) else None

    def local_grid_to_world(self, grid_pos: Optional[Tuple], base_world_pos: Optional[Tuple] = None, offset_m: Optional[Tuple] = None) -> Optional[Tuple[float, float, float]]:
        if base_world_pos is not None and offset_m is not None:
            # New functionality for RRT* steering: Apply a meter-based offset to a world coordinate.
            base_lon, base_lat, base_alt = base_world_pos
            dx_m, dy_m, dz_m = offset_m
            
            # Use the original (non-local grid) meter conversion factors
            lon_offset = dx_m / self.lon_deg_to_m
            lat_offset = dy_m / self.lat_deg_to_m
            
            new_lon = base_lon + lon_offset
            new_lat = base_lat + lat_offset
            new_alt = base_alt + dz_m
            return new_lon, new_lat, new_alt

        # Original functionality for D* Lite
        if grid_pos is None: return None
        gx, gy, gz = grid_pos
        origin_lon, origin_lat, _ = self.local_grid_origin_world
        dx_m = gx * GRID_RESOLUTION_M
        dy_m = gy * GRID_RESOLUTION_M
        new_lon = origin_lon + (dx_m / self.lon_deg_to_m)
        new_lat = origin_lat + (dy_m / self.lat_deg_to_m)
        new_alt = self.alt_min + (gz * GRID_VERTICAL_RESOLUTION_M)
        return new_lon, new_lat, new_alt

    def is_valid_local_grid_pos(self, pos: Tuple) -> bool:
        x, y, z = pos
        half_width = self.grid_width // 2
        half_height = self.grid_height // 2
        return (-half_width <= x < half_width and -half_height <= y < half_height and 0 <= z < self.grid_depth)