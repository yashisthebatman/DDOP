# path_planner.py (Full Updated Code)
import numpy as np
import logging
from typing import Tuple, List, Optional
from itertools import product
import time

import config
from environment import Environment
from ml_predictor.predictor import EnergyTimePredictor
from utils.heuristics import TimeHeuristic, EnergyHeuristic, BalancedHeuristic
from utils.d_star_lite import DStarLite # <-- Import the new D* Lite engine
from utils.geometry import calculate_wind_effect

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
WorldCoord = Tuple[float, float, float]
GridCoord = Tuple[int, int, int]

class PathPlanner3D:
    def __init__(self, env: Environment, predictor: EnergyTimePredictor):
        self.env = env
        self.predictor = predictor
        self.resolution = 15
        self.grid_shape, self.origin_lon, self.origin_lat = self._get_grid_params()

        # --- Phase 1: Sparse Grid (Voxel Hash Map) ---
        # Replaced the dense NumPy grid with a sparse dictionary for costs.
        self.cost_map = {}
        
        # --- Phase 2: Pre-computation of Costs ---
        self._precompute_cost_map()

        self.d_star_instance: Optional[DStarLite] = None
        self._calculate_heuristic_baselines()
        logging.info(f"Sparse D* Lite Planner ready. Pre-computed {len(self.cost_map)} cell costs.")

    def _precompute_cost_map(self):
        """
        Pre-calculates all environmental and obstacle costs into a sparse map.
        This is a one-time operation at startup.
        """
        logging.info("Starting one-time cost map pre-computation...")
        start_time = time.time()

        # Optional: Pre-compute wind cost layer
        if config.PRECOMPUTE_WIND_COSTS:
            logging.info("Pre-computing wind cost layer for the entire flight volume...")
            # We iterate through a reasonable flight volume, not every single cell up to max altitude
            min_alt_grid = int(config.MIN_ALTITUDE / self.resolution)
            max_alt_grid = int(config.DEFAULT_CRUISING_ALTITUDE / self.resolution) + 10 # Buffer
            
            for x in range(self.grid_shape[0]):
                for y in range(self.grid_shape[1]):
                    # Get a representative wind vector for this (x,y) column
                    world_lon, world_lat, _ = self._grid_to_world((x, y, 0))
                    wind_vector = self.env.weather.get_wind_at_location(world_lon, world_lat)
                    
                    # Assume a generic horizontal flight vector to calculate impact
                    flight_vector = np.array([1, 0, 0]) * config.DRONE_SPEED_MPS
                    time_impact, _ = calculate_wind_effect(flight_vector, wind_vector, config.DRONE_SPEED_MPS)
                    
                    cost = max(config.DEFAULT_CELL_COST, time_impact) # Ensure cost is at least 1
                    
                    for z in range(min_alt_grid, max_alt_grid):
                        self.cost_map[(x, y, z)] = cost
        
        # Pre-compute obstacle layer (infinite cost)
        logging.info("Computing obstacle locations (buildings and NFZs)...")
        # Buildings
        for b in self.env.buildings:
            min_c = self._world_to_grid((b.center_xy[0]-b.size_xy[0]/2, b.center_xy[1]-b.size_xy[1]/2, 0))
            max_c = self._world_to_grid((b.center_xy[0]+b.size_xy[0]/2, b.center_xy[1]+b.size_xy[1]/2, b.height))
            for x in range(min_c[0], max_c[0] + 1):
                for y in range(min_c[1], max_c[1] + 1):
                    for z in range(max_c[2] + 1):
                        self.cost_map[(x, y, z)] = float('inf')
        # No-Fly Zones
        for zone in self.env.static_nfzs:
            min_c = self._world_to_grid((zone[0], zone[1], 0))
            max_c = self._world_to_grid((zone[2], zone[3], config.MAX_ALTITUDE))
            for x in range(min_c[0], max_c[0] + 1):
                for y in range(min_c[1], max_c[1] + 1):
                    for z in range(self.grid_shape[2]):
                        self.cost_map[(x, y, z)] = float('inf')

        duration = time.time() - start_time
        logging.info(f"Cost map pre-computation finished in {duration:.2f} seconds.")

    def find_path(self, start_pos: WorldCoord, end_pos: WorldCoord, payload_kg: float, optimization_mode: str, time_weight: float = 0.5) -> Tuple[Optional[List[WorldCoord]], str]:
        """
        Finds the initial path by creating and running a D* Lite instance.
        """
        start_grid = self._find_nearest_valid_node(self._world_to_grid(start_pos))
        end_grid = self._find_nearest_valid_node(self._world_to_grid(end_pos))

        if not start_grid or not end_grid:
            return None, "Error: Start or End point is in an obstructed area."

        if optimization_mode == "time": heuristic_class = TimeHeuristic
        elif optimization_mode == "energy": heuristic_class = EnergyHeuristic
        else: heuristic_class = BalancedHeuristic
        heuristic = heuristic_class(self, payload_kg, end_grid, time_weight)
        
        self.d_star_instance = DStarLite(start_grid, end_grid, self.cost_map, heuristic)
        path_grid = self.d_star_instance.compute_shortest_path()

        if not path_grid:
            return None, "D* Lite initial planning failed to find a path."
        
        world_path = [self._grid_to_world(p) for p in path_grid]
        return world_path, "Initial path found successfully"

    def replan_path(self, changed_nfz: list, new_start_pos: WorldCoord) -> Tuple[Optional[List[WorldCoord]], str]:
        """
        Performs an efficient replan using the existing D* Lite instance.
        """
        if not self.d_star_instance:
            return None, "Error: Must call find_path() before replanning."

        logging.info("D* Lite: Replanning due to environmental change...")
        
        # 1. Get cost updates for the new obstacle
        changed_cells = self._get_grid_cells_in_nfz(changed_nfz)
        cost_updates = [(cell, float('inf')) for cell in changed_cells]

        # 2. Update the master cost map
        for cell, cost in cost_updates:
            self.cost_map[cell] = cost

        # 3. Call the D* Lite engine's replan method
        new_start_grid = self._world_to_grid(new_start_pos)
        path_grid = self.d_star_instance.update_and_replan(new_start_grid, cost_updates)
        
        if not path_grid:
            return None, "D* Lite replanning failed to find a valid path."
            
        world_path = [self._grid_to_world(p) for p in path_grid]
        return world_path, "Replanning successful"
    
    # --- Helper Methods ---
    def _find_nearest_valid_node(self, coord: GridCoord) -> Optional[GridCoord]:
        if coord not in self.cost_map or self.cost_map[coord] != float('inf'):
            return coord
        q = [coord]; visited = {coord}
        moves = list(product([-1,0,1],repeat=3)); moves.remove((0,0,0))
        while q:
            curr = q.pop(0)
            for move in moves:
                neighbor = (curr[0]+move[0], curr[1]+move[1], curr[2]+move[2])
                if neighbor not in self.cost_map or self.cost_map[neighbor] != float('inf'):
                    return neighbor
                if neighbor not in visited: visited.add(neighbor); q.append(neighbor)
        return None
        
    def _get_grid_cells_in_nfz(self, zone: list) -> list:
        min_c=self._world_to_grid((zone[0],zone[1],0));max_c=self._world_to_grid((zone[2],zone[3],config.MAX_ALTITUDE))
        return [(x,y,z) for x in range(min_c[0],max_c[0]+1) for y in range(min_c[1],max_c[1]+1) for z in range(self.grid_shape[2])]

    def _calculate_heuristic_baselines(self):# ... (Unchanged)
        self.baseline_time_per_meter=1.0/config.DRONE_SPEED_MPS;p1=(self.origin_lon,self.origin_lat,100);p2_lon=self.origin_lon+self.resolution/(111000*np.cos(np.radians(self.origin_lat)));p2=(p2_lon,self.origin_lat,100);_,e=self.predictor.predict(p1,p2,0,np.array([0,0,0]));self.baseline_energy_per_meter=(e/self.resolution)if e>0 else 0.005
    def _get_grid_params(self):# ... (Unchanged)
        lon0,lat0=config.AREA_BOUNDS[0],config.AREA_BOUNDS[1];w=(config.AREA_BOUNDS[2]-lon0)*111000*np.cos(np.radians(lat0));h=(config.AREA_BOUNDS[3]-lat0)*111000;x,y,z=int(w/self.resolution)+1,int(h/self.resolution)+1,int(config.MAX_ALTITUDE/self.resolution)+1;return (x,y,z),lon0,lat0
    def _world_to_grid(self, pos):# ... (Unchanged)
        x_m=(pos[0]-self.origin_lon)*111000*np.cos(np.radians(self.origin_lat));y_m=(pos[1]-self.origin_lat)*111000;c=np.clip(np.array([x_m/self.resolution,y_m/self.resolution,pos[2]/self.resolution]),0,np.array(self.grid_shape)-1);return tuple(map(int, c))
    def _grid_to_world(self, g_pos):# ... (Unchanged)
        x,y,z=g_pos[0]*self.resolution,g_pos[1]*self.resolution,g_pos[2]*self.resolution;lon=self.origin_lon+x/(111000*np.cos(np.radians(self.origin_lat)));lat=self.origin_lat+y/111000;return(lon,lat,z)