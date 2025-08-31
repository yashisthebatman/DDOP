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
        # --- PERFORMANCE OPTIMIZATION: Increased resolution to reduce grid size by 8x ---
        self.resolution = 10 # Was 5. This is a major speed and memory improvement.
        self.grid_shape, self.origin_lon, self.origin_lat = self._get_grid_params()
        self.grid = self._create_and_populate_grid()
        self.node_map, self.reverse_node_map = self._create_node_maps()
        self.moves = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1] if not (dx == 0 and dy == 0 and dz == 0)]
        
        try:
            with open("quantum_heuristic.pkl", "rb") as f:
                self.heuristic_lookup_table = pickle.load(f)
            self.waypoints_world = WAYPOINTS
            self.waypoint_names = list(self.waypoints_world.keys())
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
                neighbor = (curr[0] + move[0], curr[1] + move[1], curr[2] + move[2])
                if not (0 <= neighbor[0] < self.grid_shape[0] and 0 <= neighbor[1] < self.grid_shape[1] and 0 <= neighbor[2] < self.grid_shape[2]): continue
                if neighbor in self.node_map: return neighbor
                if neighbor not in visited: visited.add(neighbor); q.append(neighbor)
        return None
        
    def _world_to_grid(self, pos):
        x_m = (pos[0] - self.origin_lon) * 111000 * np.cos(np.radians(self.origin_lat))
        y_m = (pos[1] - self.origin_lat) * 111000
        grid_pos_np = np.array([x_m / self.resolution, y_m / self.resolution, pos[2] / self.resolution], dtype=np.int64)
        clipped_pos = np.clip(grid_pos_np, 0, np.array(self.grid_shape) - 1)
        return tuple(map(int, clipped_pos))

    def _grid_to_world(self, grid_pos):
        x_m, y_m, z_m = grid_pos[0] * self.resolution, grid_pos[1] * self.resolution, grid_pos[2] * self.resolution
        lon = self.origin_lon + x_m / (111000 * np.cos(np.radians(self.origin_lat))); lat = self.origin_lat + y_m / 111000
        return (lon, lat, z_m)

    def _calculate_path_cost(self, world_path, payload_kg, weights, use_zero_wind=False):
        if not world_path or len(world_path) < 2: return 0.0
        total_time, total_energy = 0, 0
        for i in range(len(world_path) - 1):
            p1, p2 = world_path[i], world_path[i+1]
            wind = np.array([0,0,0]) if use_zero_wind else self.env.weather.get_wind_at_location(p1[0], p1[1])
            t, e = self.predictor.predict(p1, p2, payload_kg, wind)
            if t == float('inf'): return float('inf')
            total_time += t; total_energy += e
        return weights['time'] * total_time + weights['energy'] * total_energy

    def find_baseline_path(self, start_pos, end_pos):
        start_coord = self._find_nearest_valid_node(self._world_to_grid(start_pos))
        end_coord = self._find_nearest_valid_node(self._world_to_grid(end_pos))
        
        if not start_coord or not end_coord: 
            logging.error(f"Could not find valid nodes for path from {start_pos} to {end_pos}")
            return None
        
        # --- PERFORMANCE OPTIMIZATION: Use a simple, fast heuristic for baseline calculation ---
        def simple_heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        path_coords = a_star_search(start_coord, end_coord, self.grid, self.moves, heuristic_func=simple_heuristic)
        
        if not path_coords:
            logging.warning(f"A* failed to find a path between {start_coord} and {end_coord}")
            return None
        return [self._grid_to_world(c) for c in path_coords]

    def _find_closest_waypoint_name(self, world_pos):
        return min(self.waypoint_names, key=lambda name: calculate_distance_3d(self.waypoints_world[name], world_pos))

    def find_path_realtime(self, start_pos, end_pos, payload_kg, weights):
        if self.heuristic_lookup_table is None:
            return None, "Error: Heuristic table not loaded."

        # 1. Identify the strategic mission from the pre-computed table
        start_wp_name = self._find_closest_waypoint_name(start_pos)
        end_wp_name = self._find_closest_waypoint_name(end_pos)

        if start_wp_name not in self.heuristic_lookup_table or end_wp_name not in self.heuristic_lookup_table[start_wp_name]:
            logging.warning(f"No strategic route from {start_wp_name} to {end_wp_name} in heuristic table. Falling back to simple A*.")
            path = self.find_baseline_path(start_pos, end_pos)
            return (path, "Fallback: Simple A*") if path else (None, "Error: Fallback A* failed.")
        
        # 2. Retrieve the QUBO-optimized sequence of waypoints
        strategic_route = self.heuristic_lookup_table[start_wp_name][end_wp_name]
        sequence = strategic_route['sequence']
        
        # Create the full list of points for our tactical planner to follow
        tactical_points = [start_pos] + [self.waypoints_world[wp] for wp in sequence[1:-1]] + [end_pos]
        
        # 3. Execute fast, tactical A* for each leg of the strategic journey
        full_realtime_path = []
        for i in range(len(tactical_points) - 1):
            p1, p2 = tactical_points[i], tactical_points[i+1]
            
            # The A* search for each leg is short and fast
            path_segment = self.find_baseline_path(p1, p2)
            
            if not path_segment:
                return None, f"Error: Real-time A* failed on leg {sequence[i]}->{sequence[i+1]}"
            
            # Stitch the path together, avoiding duplicate points
            full_realtime_path.extend(path_segment if i == 0 else path_segment[1:])

        return full_realtime_path, "Optimal (QUBO Strategic + A* Tactical)"