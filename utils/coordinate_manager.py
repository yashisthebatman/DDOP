# ==============================================================================
# File: utils/coordinate_manager.py (Phase 2: Algorithm Stability)
# ==============================================================================

import numpy as np
import logging

class CoordinateManager:
    def __init__(self, grid_shape=(100, 100, 100), grid_depth=100, margin=1):
        self.grid_shape = grid_shape
        self.grid_depth = grid_depth
        self.margin = margin

    def float_to_grid(self, pos):
        x, y, z = pos
        grid = (int(round(x)), int(round(y)), int(round(z)))
        return self.round_and_validate_grid(grid)

    def grid_to_world(self, grid):
        return tuple(float(c) for c in grid)

    def round_and_validate_grid(self, grid):
        x, y, z = grid
        x = int(round(x))
        y = int(round(y))
        z = int(round(z))
        x = max(self.margin, min(x, self.grid_shape[0] - 1 - self.margin))
        y = max(self.margin, min(y, self.grid_shape[1] - 1 - self.margin))
        z = max(self.margin, min(z, self.grid_shape[2] - 1 - self.margin))
        return (x, y, z)

    def is_valid_grid_position(self, pos):
        x, y, z = pos
        return (
            self.margin <= x < self.grid_shape[0] - self.margin and
            self.margin <= y < self.grid_shape[1] - self.margin and
            self.margin <= z < self.grid_shape[2] - self.margin
        )

    def safe_grid_conversion(self, pos):
        if isinstance(pos, (list, tuple)) and len(pos) == 3:
            grid = self.float_to_grid(pos)
            if self.is_valid_grid_position(grid):
                return grid
            else:
                logging.warning(f"CoordinateManager: Position {pos} out of grid bounds.")
                return None
        else:
            logging.error(f"CoordinateManager: Invalid position format for grid conversion: {pos}")
            return None

    def find_nearest_valid_grid_cell(self, pos, max_radius=None):
        start = self.float_to_grid(pos)
        if self.is_valid_grid_position(start):
            return start
        max_radius = max_radius or max(self.grid_shape)
        for radius in range(1, max_radius):
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    for dz in range(-radius, radius+1):
                        candidate = (start[0]+dx, start[1]+dy, start[2]+dz)
                        if self.is_valid_grid_position(candidate):
                            return candidate
        logging.error(f"CoordinateManager: No valid grid cell found near {pos} within radius {max_radius}")
        return None
