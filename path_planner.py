# ==============================================================================
# path_planner.py
# ==============================================================================
import numpy as np
import logging
import pickle


# --- Assume these are imported from your project structure ---
from config import AREA_BOUNDS, MAX_ALTITUDE, MIN_ALTITUDE, WAYPOINTS
from utils.heuristics import a_star_search
from utils.geometry import calculate_distance_3d
# -----------------------------------------------------------


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PathPlanner3D:
    def __init__(self, env, predictor):
        
        self.env = env
        self.predictor = predictor
        self.resolution = 5
        self.grid_shape, self.origin_lon, self.origin_lat = self._get_grid_params()
        self.grid = self._create_and_populate_grid()
        self.node_map, self.reverse_node_map = self._create_node_maps()
        self.moves = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1] if not (dx == 0 and dy == 0 and dz == 0)]
        
        try:
            with open("quantum_heuristic.pkl", "rb") as f:
                self.heuristic_lookup_table = pickle.load(f)
            self.waypoints_grid = {name: self._world_to_grid(pos) for name, pos in WAYPOINTS.items()}
            self.waypoint_names = list(self.waypoints_grid.keys())
            logging.info("Successfully loaded pre-computed 'quantum_heuristic.pkl'. Planner is in ONLINE mode.")
        except FileNotFoundError:
            logging.warning("Heuristic file 'quantum_heuristic.pkl' not found. Planner is in OFFLINE mode (for generation only).")
            self.heuristic_lookup_table = None

    def _get_grid_params(self):
        origin_lon, origin_lat = AREA_BOUNDS[0], AREA_BOUNDS[1]
        width_m = (AREA_BOUNDS[2] - origin_lon) * 111000 * np.cos(np.radians(origin_lat)); height_m = (AREA_BOUNDS[3] - origin_lat) * 111000
        x_dim, y_dim = int(width_m / self.resolution) + 1, int(height_m / self.resolution) + 1; z_dim = int(MAX_ALTITUDE / self.resolution) + 1
        return (x_dim, y_dim, z_dim), origin_lon, origin_lat

    def _create_and_populate_grid(self):
        grid = np.zeros(self.grid_shape, dtype=np.uint8); min_alt_grid = int(MIN_ALTITUDE / self.resolution); grid[:, :, :min_alt_grid] = 1
        for b in self.env.buildings:
            min_c = self._world_to_grid((b.center_xy[0] - b.size_xy[0]/2, b.center_xy[1] - b.size_xy[1]/2, 0)); max_c = self._world_to_grid((b.center_xy[0] + b.size_xy[0]/2, b.center_xy[1] + b.size_xy[1]/2, b.height))
            grid[min_c[0]:max_c[0]+1, min_c[1]:max_c[1]+1, :max_c[2]+1] = 1
        for zone in self.env.no_fly_zones:
            min_c = self._world_to_grid((zone[0], zone[1], 0)); max_c = self._world_to_grid((zone[2], zone[3], MAX_ALTITUDE))
            grid[min_c[0]:max_c[0]+1, min_c[1]:max_c[1]+1, :] = 1
        return grid

    def _create_node_maps(self):
        node_map, reverse_node_map = {}, []; idx = 0
        for x in range(self.grid_shape[0]):
            for y in range(self.grid_shape[1]):
                for z in range(self.grid_shape[2]):
                    if self.grid[x, y, z] == 0: coord = (x, y, z); node_map[coord] = idx; reverse_node_map.append(coord); idx += 1
        return node_map, reverse_node_map

    def _find_nearest_valid_node(self, coord):
        if coord in self.node_map: return coord
        q = [coord]; visited = {coord}
        while q:
            curr = q.pop(0)
            for move in self.moves:
                # --- FINAL FIX: Use pure Python math to avoid creating numpy integers ---
                neighbor = (curr[0] + move[0], curr[1] + move[1], curr[2] + move[2])
                # -------------------------------------------------------------------------
                if not (0 <= neighbor[0] < self.grid_shape[0] and 0 <= neighbor[1] < self.grid_shape[1] and 0 <= neighbor[2] < self.grid_shape[2]): continue
                if neighbor in self.node_map: return neighbor
                if neighbor not in visited: visited.add(neighbor); q.append(neighbor)
        return None
        
    def _world_to_grid(self, pos):
        """Converts world coordinates to grid coordinates."""
        x_m = (pos[0] - self.origin_lon) * 111000 * np.cos(np.radians(self.origin_lat))
        y_m = (pos[1] - self.origin_lat) * 111000
        
        grid_pos_np = np.array([
            x_m / self.resolution,
            y_m / self.resolution,
            pos[2] / self.resolution
        ], dtype=np.int64)
        
        clipped_pos = np.clip(grid_pos_np, 0, np.array(self.grid_shape) - 1)
        
        # This fix remains correct and necessary
        return tuple(map(int, clipped_pos))

    # --- (The rest of the file is unchanged and correct) ---
    def _grid_to_world(self, grid_pos):
        x_m, y_m, z_m = grid_pos[0] * self.resolution, grid_pos[1] * self.resolution, grid_pos[2] * self.resolution
        lon = self.origin_lon + x_m / (111000 * np.cos(np.radians(self.origin_lat))); lat = self.origin_lat + y_m / 111000
        return (lon, lat, z_m)

    def _calculate_path_cost(self, world_path, payload_kg, weights, use_zero_wind=False):
        if not world_path or len(world_path) < 2:
            return 0.0
        total_time, total_energy = 0, 0
        for i in range(len(world_path) - 1):
            p1, p2 = world_path[i], world_path[i+1]
            wind = np.array([0,0,0]) if use_zero_wind else self.env.weather.get_wind_at_location(p1[0], p1[1])
            t, e = self.predictor.predict(p1, p2, payload_kg, wind)
            if t == float('inf'): return float('inf')
            total_time += t; total_energy += e
        return weights['time'] * total_time + weights['energy'] * total_energy

    def find_baseline_path(self, start_pos, end_pos, payload_kg, weights):
        start_coord_raw = self._world_to_grid(start_pos); end_coord_raw = self._world_to_grid(end_pos)
        start_coord = self._find_nearest_valid_node(start_coord_raw); end_coord = self._find_nearest_valid_node(end_coord_raw)
        
        if not start_coord or not end_coord: 
            logging.error(f"Could not find valid nodes for path from {start_pos} to {end_pos}")
            return None
            
        if start_coord == end_coord: 
            return [self._grid_to_world(start_coord)] 
            
        path_coords = a_star_search(start_coord, end_coord, self.grid, self.moves)
        
        if not path_coords:
            logging.warning(f"A* failed to find a path between {start_coord} and {end_coord}")
            return None
            
        return [self._grid_to_world(c) for c in path_coords]

    def _calculate_wind_impact_on_path(self, path, payload_kg, weights):
        cost_with_wind = self._calculate_path_cost(path, payload_kg, weights, use_zero_wind=False)
        cost_no_wind = self._calculate_path_cost(path, payload_kg, weights, use_zero_wind=True)
        return cost_with_wind - cost_no_wind if cost_with_wind != float('inf') else float('inf')

    def _find_closest_waypoint_name(self, grid_coord):
        return min(self.waypoint_names, key=lambda name: np.linalg.norm(np.array(self.waypoints_grid[name]) - np.array(grid_coord)))

    def find_path_realtime(self, start_pos, end_pos, payload_kg, weights):
        if self.heuristic_lookup_table is None:
            logging.error("Cannot find path: Heuristic table not loaded.")
            return None, "Error: Heuristic table not loaded."
        start_coord_raw = self._world_to_grid(start_pos); end_coord_raw = self._world_to_grid(end_pos)
        start_coord = self._find_nearest_valid_node(start_coord_raw); end_coord = self._find_nearest_valid_node(end_coord_raw)
        if not start_coord or not end_coord: return None, "Error: Start or End point is in an invalid area."
        def dynamic_heuristic(current_grid_pos, end_grid_pos):
            start_wp_name = self._find_closest_waypoint_name(current_grid_pos)
            end_wp_name = self._find_closest_waypoint_name(end_grid_pos)
            if start_wp_name in self.heuristic_lookup_table and end_wp_name in self.heuristic_lookup_table[start_wp_name]:
                baseline_cost, baseline_path = self.heuristic_lookup_table[start_wp_name][end_wp_name]
                wind_correction = self._calculate_wind_impact_on_path(baseline_path, payload_kg, weights)
                return baseline_cost + wind_correction
            return calculate_distance_3d(self._grid_to_world(current_grid_pos), self._grid_to_world(end_grid_pos))
        
        path_coords = a_star_search(start_coord, end_coord, self.grid, self.moves, heuristic_func=dynamic_heuristic)

        if not path_coords: return None, "Error: A* failed to find a valid path"
        return [self._grid_to_world(c) for c in path_coords], "Real-time Optimal (Heuristic Guided)"