# path_planner.py
import numpy as np
import logging
from typing import Tuple, List, Optional
from itertools import product
from multiprocessing import Pool, cpu_count
from numba import njit, prange

import config
from environment import Environment
from ml_predictor.predictor import EnergyTimePredictor
from utils.a_star import a_star_search
from utils.heuristics import TimeHeuristic, EnergyHeuristic, BalancedHeuristic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
WorldCoord = Tuple[float, float, float]
GridCoord = Tuple[int, int, int]

# --- Numba JIT Compiled Functions for Speed ---

@njit
def clip_scalar(val, min_val, max_val):
    """A Numba-compatible scalar clip function."""
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val

@njit(parallel=True)
def populate_grid_numba(grid, buildings_data, nfzs_data, min_alt_grid, world_to_grid_params):
    """A Numba-accelerated function to populate the grid with obstacles."""
    shape = grid.shape
    origin_lon, origin_lat, resolution, cos_lat = world_to_grid_params
    
    # Vectorized calculation for world to grid conversion inside Numba
    def world_to_grid_internal(lon, lat, alt):
        x_m = (lon - origin_lon) * 111000 * cos_lat
        y_m = (lat - origin_lat) * 111000
        
        # --- MODIFICATION START ---
        # Use the Numba-compatible scalar clip function instead of np.clip
        gx_float = x_m / resolution
        gy_float = y_m / resolution
        gz_float = alt / resolution
        
        gx = int(clip_scalar(gx_float, 0, shape[0] - 1))
        gy = int(clip_scalar(gy_float, 0, shape[1] - 1))
        gz = int(clip_scalar(gz_float, 0, shape[2] - 1))
        # --- MODIFICATION END ---
        
        return gx, gy, gz

    grid[:, :, :min_alt_grid] = 1 # Mark below-min-altitude
    
    for i in prange(len(buildings_data)):
        b = buildings_data[i]
        min_c = world_to_grid_internal(b[0] - b[2]/2, b[1] - b[3]/2, 0)
        max_c = world_to_grid_internal(b[0] + b[2]/2, b[1] + b[3]/2, b[4])
        grid[min_c[0]:max_c[0]+1, min_c[1]:max_c[1]+1, :max_c[2]+1] = 1
            
    for i in prange(len(nfzs_data)):
        zone = nfzs_data[i]
        min_c = world_to_grid_internal(zone[0], zone[1], 0)
        max_c = world_to_grid_internal(zone[2], zone[3], config.MAX_ALTITUDE)
        grid[min_c[0]:max_c[0]+1, min_c[1]:max_c[1]+1, :] = 1
    return grid

# --- Worker for Multiprocessing ---
# ... (This part is unchanged) ...
def find_path_segment_worker(args):
    """Worker function to find a path for a single segment in parallel."""
    segment_start, segment_end, grid, moves, heuristic_class, planner_instance, payload_kg, time_weight = args
    heuristic = heuristic_class(planner_instance, payload_kg, segment_end, time_weight)
    return a_star_search(segment_start, segment_end, grid, moves, heuristic)


class PathPlanner3D:
    # ... (The rest of the PathPlanner3D class is unchanged) ...
    def __init__(self, env: Environment, predictor: EnergyTimePredictor):
        self.env = env
        self.predictor = predictor
        self.resolution = 15
        self.coarse_factor = 5
        self.grid_shape, self.origin_lon, self.origin_lat = self._get_grid_params()
        self.grid = self.create_grid()
        self.moves = list(product([-1, 0, 1], repeat=3)); self.moves.remove((0, 0, 0))
        self._calculate_heuristic_baselines()
        logging.info(f"HPA* PathPlanner initialized. Fine Grid: {self.grid.shape}")

    def _calculate_heuristic_baselines(self):
        self.baseline_time_per_meter = 1.0 / config.DRONE_SPEED_MPS
        p1 = (self.origin_lon, self.origin_lat, 100)
        p2_lon = self.origin_lon + self.resolution / (111000 * np.cos(np.radians(self.origin_lat)))
        p2 = (p2_lon, self.origin_lat, 100)
        _, energy_cost = self.predictor.predict(p1, p2, 0, np.array([0,0,0]))
        self.baseline_energy_per_meter = (energy_cost / self.resolution) if energy_cost > 0 else 0.005


    def create_grid(self, dynamic_nfzs=None):
        """Creates the 3D grid, now accelerated by Numba."""
        if dynamic_nfzs is None: dynamic_nfzs = []
        grid = np.zeros(self.grid_shape, dtype=np.uint8)
        min_alt_grid = int(config.MIN_ALTITUDE / self.resolution)

        # Prepare data for Numba function
        buildings_data = np.array([(b.center_xy[0], b.center_xy[1], b.size_xy[0], b.size_xy[1], b.height) for b in self.env.buildings])
        all_nfzs = self.env.static_nfzs + dynamic_nfzs
        nfzs_data = np.array(all_nfzs)
        
        world_to_grid_params = (self.origin_lon, self.origin_lat, self.resolution, np.cos(np.radians(self.origin_lat)))
        
        return populate_grid_numba(grid, buildings_data, nfzs_data, min_alt_grid, world_to_grid_params)

    def find_path(self, start_pos: WorldCoord, end_pos: WorldCoord, payload_kg: float, optimization_mode: str, time_weight: float = 0.5) -> Tuple[Optional[List[WorldCoord]], str]:
        current_grid = self.create_grid(self.env.dynamic_nfzs)
        
        start_grid = self._find_nearest_valid_node(self._world_to_grid(start_pos), current_grid)
        end_grid = self._find_nearest_valid_node(self._world_to_grid(end_pos), current_grid)

        if not start_grid or not end_grid:
            return None, "Error: Start or End point is in an invalid/obstructed area."

        alt = config.DEFAULT_CRUISING_ALTITUDE
        mid_lon = (start_pos[0] + end_pos[0]) / 2
        mid_lat = (start_pos[1] + end_pos[1]) / 2
        mid_point_world = (mid_lon, mid_lat, alt)
        mid_point_grid = self._find_nearest_valid_node(self._world_to_grid(mid_point_world), current_grid)
        
        waypoints = [start_grid, mid_point_grid, end_grid] if mid_point_grid else [start_grid, end_grid]
        waypoints = sorted(set(waypoints), key=waypoints.index) # Remove duplicates

        if optimization_mode == "time": heuristic_class = TimeHeuristic
        elif optimization_mode == "energy": heuristic_class = EnergyHeuristic
        else: heuristic_class = BalancedHeuristic

        tasks = []
        for i in range(len(waypoints) - 1):
            tasks.append((waypoints[i], waypoints[i+1], current_grid, self.moves, heuristic_class, self, payload_kg, time_weight))

        full_fine_path = []
        try:
            with Pool(processes=cpu_count()) as pool:
                logging.info(f"Calculating {len(tasks)} path segments in parallel on {cpu_count()} cores...")
                results = pool.map(find_path_segment_worker, tasks)
            
            for i, segment_path in enumerate(results):
                if not segment_path:
                    return None, f"HPA* Error: Failed to connect waypoints {i} and {i+1}."
                full_fine_path.extend(segment_path if i == 0 else segment_path[1:])
        except Exception as e:
            logging.error(f"Multiprocessing failed: {e}")
            return None, "Error during parallel pathfinding."

        if not full_fine_path: return None, "HPA* Error: Path refinement failed."
        world_path = [self._grid_to_world(p) for p in full_fine_path]
        return world_path, "Path found successfully"

    def _get_grid_params(self) -> Tuple[GridCoord, float, float]:
        origin_lon, origin_lat = config.AREA_BOUNDS[0], config.AREA_BOUNDS[1]
        width_m = (config.AREA_BOUNDS[2] - origin_lon) * 111000 * np.cos(np.radians(origin_lat))
        height_m = (config.AREA_BOUNDS[3] - origin_lat) * 111000
        x_dim = int(width_m / self.resolution) + 1
        y_dim = int(height_m / self.resolution) + 1
        z_dim = int(config.MAX_ALTITUDE / self.resolution) + 1
        return (x_dim, y_dim, z_dim), origin_lon, origin_lat
        
    def _world_to_grid(self, pos: WorldCoord) -> GridCoord:
        x_m = (pos[0] - self.origin_lon) * 111000 * np.cos(np.radians(self.origin_lat))
        y_m = (pos[1] - self.origin_lat) * 111000
        grid_pos = np.array([x_m / self.resolution, y_m / self.resolution, pos[2] / self.resolution])
        clipped = np.clip(grid_pos, 0, np.array(self.grid_shape) - 1)
        return tuple(map(int, clipped))

    def _grid_to_world(self, grid_pos: GridCoord) -> WorldCoord:
        x_m = grid_pos[0] * self.resolution
        y_m = grid_pos[1] * self.resolution
        z_m = grid_pos[2] * self.resolution
        lon = self.origin_lon + x_m / (111000 * np.cos(np.radians(self.origin_lat)))
        lat = self.origin_lat + y_m / 111000
        return (lon, lat, z_m)

    def _find_nearest_valid_node(self, coord: GridCoord, grid: np.ndarray) -> Optional[GridCoord]:
        if grid[coord] == 0: return coord
        q = [coord]; visited = {coord}
        shape = grid.shape
        while q:
            curr = q.pop(0)
            for move in self.moves:
                neighbor = (curr[0] + move[0], curr[1] + move[1], curr[2] + move[2])
                if not (0 <= neighbor[0] < shape[0] and 0 <= neighbor[1] < shape[1] and 0 <= neighbor[2] < shape[2]):
                    continue
                if grid[neighbor] == 0: return neighbor
                if neighbor not in visited: visited.add(neighbor); q.append(neighbor)
        return None