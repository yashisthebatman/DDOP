import numpy as np
import logging
from typing import Tuple, List, Optional

import config
from environment import Environment
from ml_predictor.predictor import EnergyTimePredictor
from utils.a_star import a_star_search
from utils.heuristics import TimeHeuristic, EnergyHeuristic, BalancedHeuristic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
WorldCoord = Tuple[float, float, float]
GridCoord = Tuple[int, int, int]

class PathPlanner3D:
    """
    The main application logic controller for pathfinding.

    This class, "Mission Control," is responsible for:
    - Managing the 3D grid representation of the environment.
    - Orchestrating the pathfinding process by selecting the correct heuristic.
    - Calling the generic A* search algorithm.
    - Converting coordinates between the real world and the grid.
    """

    def __init__(self, env: Environment, predictor: EnergyTimePredictor):
        self.env = env
        self.predictor = predictor
        self.resolution = 10  # meters per grid cell
        self.grid_shape, self.origin_lon, self.origin_lat = self._get_grid_params()
        self.grid = self._create_and_populate_grid()
        self.moves = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1] if not (dx == 0 and dy == 0 and dz == 0)]
        self._calculate_heuristic_baselines()
        logging.info("PathPlanner3D initialized with intelligent heuristic engine.")

    def _calculate_heuristic_baselines(self):
        """Calculates baseline values used for fast, effective heuristic estimates."""
        self.baseline_time_per_meter = 1.0 / config.DRONE_SPEED_MPS

        p1 = (self.origin_lon, self.origin_lat, 100)
        p2 = (self.origin_lon + 1 / (111000 * np.cos(np.radians(self.origin_lat))), self.origin_lat, 100)
        _, energy_cost = self.predictor.predict(p1, p2, 0, np.array([0,0,0]))
        self.baseline_energy_per_meter = energy_cost if energy_cost > 0 else 0.005 # Fallback
        
        logging.info(f"Heuristic baselines: time_per_m={self.baseline_time_per_meter:.4f}, energy_per_m={self.baseline_energy_per_meter:.4f}")

    def find_path(self, start_pos: WorldCoord, end_pos: WorldCoord, payload_kg: float, optimization_mode: str) -> Tuple[Optional[List[WorldCoord]], str]:
        """
        The main public interface for finding a path.
        
        Args:
            start_pos: World coordinates (lon, lat, alt) for the start.
            end_pos: World coordinates for the end.
            payload_kg: The drone's current payload.
            optimization_mode: The goal ('time', 'energy', or 'balanced').

        Returns:
            A tuple of (path, status). Path is None if not found.
        """
        start_grid = self._find_nearest_valid_node(self._world_to_grid(start_pos))
        end_grid = self._find_nearest_valid_node(self._world_to_grid(end_pos))

        if not start_grid or not end_grid:
            return None, "Error: Start or End point is in an invalid/obstructed area."

        # 1. Select the appropriate heuristic based on the optimization mode
        if optimization_mode == "time":
            heuristic = TimeHeuristic(self, payload_kg, end_grid)
        elif optimization_mode == "energy":
            heuristic = EnergyHeuristic(self, payload_kg, end_grid)
        else: # "balanced"
            heuristic = BalancedHeuristic(self, payload_kg, end_grid)

        # 2. Call the generic A* search algorithm
        path_grid = a_star_search(start_grid, end_grid, self.grid, self.moves, heuristic)

        # 3. Process and return the result
        if path_grid:
            world_path = [self._grid_to_world(p) for p in path_grid]
            return world_path, "Path found successfully"
        else:
            return None, "Error: A* failed to find a valid path"

    # --- Grid Management and Coordinate Conversion Helpers ---
    def _get_grid_params(self) -> Tuple[GridCoord, float, float]:
        origin_lon, origin_lat = config.AREA_BOUNDS[0], config.AREA_BOUNDS[1]
        width_m = (config.AREA_BOUNDS[2] - origin_lon) * 111000 * np.cos(np.radians(origin_lat))
        height_m = (config.AREA_BOUNDS[3] - origin_lat) * 111000
        x_dim = int(width_m / self.resolution) + 1
        y_dim = int(height_m / self.resolution) + 1
        z_dim = int(config.MAX_ALTITUDE / self.resolution) + 1
        return (x_dim, y_dim, z_dim), origin_lon, origin_lat

    def _create_and_populate_grid(self) -> np.ndarray:
        grid = np.zeros(self.grid_shape, dtype=np.uint8)
        min_alt_grid = int(config.MIN_ALTITUDE / self.resolution)
        grid[:, :, :min_alt_grid] = 1 # Mark below-min-altitude as obstacle
        
        for b in self.env.buildings:
            min_c = self._world_to_grid((b.center_xy[0] - b.size_xy[0]/2, b.center_xy[1] - b.size_xy[1]/2, 0))
            max_c = self._world_to_grid((b.center_xy[0] + b.size_xy[0]/2, b.center_xy[1] + b.size_xy[1]/2, b.height))
            grid[min_c[0]:max_c[0]+1, min_c[1]:max_c[1]+1, :max_c[2]+1] = 1
            
        for zone in config.NO_FLY_ZONES:
            min_c = self._world_to_grid((zone[0], zone[1], 0))
            max_c = self._world_to_grid((zone[2], zone[3], config.MAX_ALTITUDE))
            grid[min_c[0]:max_c[0]+1, min_c[1]:max_c[1]+1, :] = 1
        return grid
        
    def _world_to_grid(self, pos: WorldCoord) -> GridCoord:
        x_m = (pos[0] - self.origin_lon) * 111000 * np.cos(np.radians(self.origin_lat))
        y_m = (pos[1] - self.origin_lat) * 111000
        grid_pos_np = np.array([x_m / self.resolution, y_m / self.resolution, pos[2] / self.resolution], dtype=np.int64)
        clipped_pos = np.clip(grid_pos_np, 0, np.array(self.grid_shape) - 1)
        return tuple(map(int, clipped_pos))

    def _grid_to_world(self, grid_pos: GridCoord) -> WorldCoord:
        x_m = grid_pos[0] * self.resolution
        y_m = grid_pos[1] * self.resolution
        z_m = grid_pos[2] * self.resolution
        lon = self.origin_lon + x_m / (111000 * np.cos(np.radians(self.origin_lat)))
        lat = self.origin_lat + y_m / 111000
        return (lon, lat, z_m)

    def _find_nearest_valid_node(self, coord: GridCoord) -> Optional[GridCoord]:
        if self.grid[coord] == 0: return coord
        q = [coord]; visited = {coord}
        while q:
            curr = q.pop(0)
            for move in self.moves:
                neighbor = (curr[0] + move[0], curr[1] + move[1], curr[2] + move[2])
                if not (0 <= neighbor[0] < self.grid_shape[0] and 0 <= neighbor[1] < self.grid_shape[1] and 0 <= neighbor[2] < self.grid_shape[2]):
                    continue
                if self.grid[neighbor] == 0: return neighbor
                if neighbor not in visited: visited.add(neighbor); q.append(neighbor)
        return None