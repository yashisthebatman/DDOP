# path_planner.py
import numpy as np
import logging
from typing import Tuple, List, Optional
from itertools import product
from numba import njit, prange

import config
from environment import Environment
from ml_predictor.predictor import EnergyTimePredictor
from utils.heuristics import TimeHeuristic, EnergyHeuristic, BalancedHeuristic
from utils.lpa_star import LPAStar  # <-- Import the new engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
WorldCoord = Tuple[float, float, float]
GridCoord = Tuple[int, int, int]

# Numba-accelerated grid population function (unchanged from previous version)
@njit
def clip_scalar(val, min_val, max_val):
    if val < min_val: return min_val
    if val > max_val: return max_val
    return val
@njit(parallel=True)
def populate_grid_numba(grid, buildings_data, nfzs_data, min_alt_grid, world_to_grid_params):
    # ... (This Numba function remains exactly the same)
    shape = grid.shape
    origin_lon, origin_lat, resolution, cos_lat = world_to_grid_params
    def world_to_grid_internal(lon, lat, alt):
        x_m = (lon - origin_lon) * 111000 * cos_lat
        y_m = (lat - origin_lat) * 111000
        gx = int(clip_scalar(x_m / resolution, 0, shape[0] - 1))
        gy = int(clip_scalar(y_m / resolution, 0, shape[1] - 1))
        gz = int(clip_scalar(alt / resolution, 0, shape[2] - 1))
        return gx, gy, gz
    grid[:, :, :min_alt_grid] = 1
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

class PathPlanner3D:
    def __init__(self, env: Environment, predictor: EnergyTimePredictor):
        self.env = env
        self.predictor = predictor
        self.resolution = 15
        self.grid_shape, self.origin_lon, self.origin_lat = self._get_grid_params()
        self.grid = self.create_grid()
        self.lpa_instance: Optional[LPAStar] = None # <-- Will hold the stateful engine
        self._calculate_heuristic_baselines()
        logging.info(f"LPA* PathPlanner initialized. Grid: {self.grid.shape}")

    def find_path(self, start_pos: WorldCoord, end_pos: WorldCoord, payload_kg: float, optimization_mode: str, time_weight: float = 0.5) -> Tuple[Optional[List[WorldCoord]], str]:
        """
        Performs the INITIAL path plan by creating and running an LPA* instance.
        """
        logging.info("LPA*: Performing initial full path computation...")
        start_grid = self._find_nearest_valid_node(self._world_to_grid(start_pos), self.grid)
        end_grid = self._find_nearest_valid_node(self._world_to_grid(end_pos), self.grid)

        if not start_grid or not end_grid:
            return None, "Error: Start or End point is in an invalid/obstructed area."

        if optimization_mode == "time": heuristic_class = TimeHeuristic
        elif optimization_mode == "energy": heuristic_class = EnergyHeuristic
        else: heuristic_class = BalancedHeuristic
        
        heuristic = heuristic_class(self, payload_kg, end_grid, time_weight)
        
        # Create and store the stateful LPA* engine instance
        self.lpa_instance = LPAStar(self.grid, start_grid, end_grid, heuristic)
        self.lpa_instance.compute_shortest_path()
        
        path_grid = self.lpa_instance.get_path()

        if not path_grid:
            return None, "LPA* initial planning failed to find a path."
        
        world_path = [self._grid_to_world(p) for p in path_grid]
        return world_path, "Initial path found successfully"

    def replan_path(self, changed_nfz: list) -> Tuple[Optional[List[WorldCoord]], str]:
        """
        Performs an efficient REPLAN using the existing LPA* instance.
        """
        if not self.lpa_instance:
            return None, "Error: Must call find_path() for initial plan before replanning."

        logging.info("LPA*: Replanning due to environmental change...")
        
        # 1. Get the list of grid cells affected by the change
        changed_cells = self._get_grid_cells_in_nfz(changed_nfz)
        
        # 2. Update the master grid and the LPA* instance's view of the grid
        for x, y, z in changed_cells:
            self.grid[x, y, z] = 1  # Mark as obstacle
        self.lpa_instance.grid = self.grid
        
        # 3. Inform LPA* about the changed cells and their neighbors
        nodes_to_update = set()
        for cell in changed_cells:
            nodes_to_update.add(cell)
            # Also update neighbors, as their path *through* the changed cell is now invalid
            for neighbor in self.lpa_instance._get_neighbors(cell):
                nodes_to_update.add(neighbor)
        
        for node in nodes_to_update:
            self.lpa_instance._update_node(node)
            
        # 4. Re-compute the path. This will be very fast.
        self.lpa_instance.compute_shortest_path()
        path_grid = self.lpa_instance.get_path()

        if not path_grid:
            return None, "LPA* replanning failed to find a valid path."
            
        world_path = [self._grid_to_world(p) for p in path_grid]
        return world_path, "Replanning successful"

    def _get_grid_cells_in_nfz(self, zone: list) -> list:
        """Helper to convert a world-space NFZ into a list of grid cells."""
        min_c = self._world_to_grid((zone[0], zone[1], 0))
        max_c = self._world_to_grid((zone[2], zone[3], config.MAX_ALTITUDE))
        cells = []
        for x in range(min_c[0], max_c[0] + 1):
            for y in range(min_c[1], max_c[1] + 1):
                for z in range(self.grid_shape[2]):
                    cells.append((x, y, z))
        return cells
        
    # --- Grid Management and Coordinate Conversion Helpers (Unchanged) ---
    def _calculate_heuristic_baselines(self):
        self.baseline_time_per_meter = 1.0 / config.DRONE_SPEED_MPS
        p1=(self.origin_lon,self.origin_lat,100);p2_lon=self.origin_lon+self.resolution/(111000*np.cos(np.radians(self.origin_lat)));p2=(p2_lon,self.origin_lat,100)
        _, e = self.predictor.predict(p1, p2, 0, np.array([0,0,0]));self.baseline_energy_per_meter=(e/self.resolution)if e>0 else 0.005
    def create_grid(self, dynamic_nfzs=None):
        if dynamic_nfzs is None: dynamic_nfzs = []
        grid = np.zeros(self.grid_shape, dtype=np.uint8)
        min_alt_grid = int(config.MIN_ALTITUDE / self.resolution)
        b_data=np.array([(b.center_xy[0],b.center_xy[1],b.size_xy[0],b.size_xy[1],b.height) for b in self.env.buildings])
        nfzs_data = np.array(self.env.static_nfzs + dynamic_nfzs)
        params=(self.origin_lon, self.origin_lat, self.resolution, np.cos(np.radians(self.origin_lat)))
        return populate_grid_numba(grid, b_data, nfzs_data, min_alt_grid, params)
    def _get_grid_params(self):
        ol,ola=config.AREA_BOUNDS[0],config.AREA_BOUNDS[1];w=(config.AREA_BOUNDS[2]-ol)*111000*np.cos(np.radians(ola));h=(config.AREA_BOUNDS[3]-ola)*111000
        x,y,z=int(w/self.resolution)+1,int(h/self.resolution)+1,int(config.MAX_ALTITUDE/self.resolution)+1;return (x,y,z),ol,ola
    def _world_to_grid(self, pos):
        x_m=(pos[0]-self.origin_lon)*111000*np.cos(np.radians(self.origin_lat));y_m=(pos[1]-self.origin_lat)*111000
        c=np.clip(np.array([x_m/self.resolution,y_m/self.resolution,pos[2]/self.resolution]),0,np.array(self.grid_shape)-1);return tuple(map(int, c))
    def _grid_to_world(self, g_pos):
        x,y,z=g_pos[0]*self.resolution,g_pos[1]*self.resolution,g_pos[2]*self.resolution
        lon=self.origin_lon+x/(111000*np.cos(np.radians(self.origin_lat)));lat=self.origin_lat+y/111000;return(lon,lat,z)
    def _find_nearest_valid_node(self, coord, grid):
        if grid[coord]==0:return coord;q=[coord];v={coord};s=grid.shape
        while q:
            c=q.pop(0)
            for m in product([-1,0,1],repeat=3):
                if m==(0,0,0):continue
                n=(c[0]+m[0],c[1]+m[1],c[2]+m[2])
                if not(0<=n[0]<s[0]and 0<=n[1]<s[1]and 0<=n[2]<s[2]):continue
                if grid[n]==0:return n
                if n not in v:v.add(n);q.append(n)
        return None