import numpy as np
import pyqubo
from dwave.samplers import SimulatedAnnealingSampler
import logging
import heapq
import itertools
import time

import config
from utils.heuristics import a_star_search
from utils.geometry import calculate_distance_3d

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PathPlanner3D:
    def __init__(self, env, predictor):
        self.env = env
        self.predictor = predictor
        self.resolution = 125
        self.grid_shape, self.origin_lon, self.origin_lat = self._get_grid_params()
        self.grid = self._create_and_populate_grid()
        self.node_map, self.reverse_node_map = self._create_node_maps()
        self.moves = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1] if not (dx == 0 and dy == 0 and dz == 0)]
        
        # --- OFFLINE PHASE ---
        self.waypoints = self._initialize_waypoints()
        self.heuristic_lookup_table = self._precompute_heuristic_lookup_table()
        logging.info("Hierarchical PathPlanner initialized with pre-computed QUBO heuristics.")

    # --- Grid and Node Setup (Unchanged) ---
    def _get_grid_params(self):
        origin_lon, origin_lat = config.AREA_BOUNDS[0], config.AREA_BOUNDS[1]
        width_m = (config.AREA_BOUNDS[2] - origin_lon) * 111000 * np.cos(np.radians(origin_lat)); height_m = (config.AREA_BOUNDS[3] - origin_lat) * 111000
        x_dim, y_dim = int(width_m / self.resolution) + 1, int(height_m / self.resolution) + 1; z_dim = int(config.MAX_ALTITUDE / self.resolution) + 1
        return (x_dim, y_dim, z_dim), origin_lon, origin_lat
    def _create_and_populate_grid(self):
        grid = np.zeros(self.grid_shape, dtype=int); min_alt_grid = int(config.MIN_ALTITUDE / self.resolution); grid[:, :, :min_alt_grid] = 1
        for b in self.env.buildings:
            min_c = self._world_to_grid((b.center_xy[0] - b.size_xy[0]/2, b.center_xy[1] - b.size_xy[1]/2, 0)); max_c = self._world_to_grid((b.center_xy[0] + b.size_xy[0]/2, b.center_xy[1] + b.size_xy[1]/2, b.height))
            grid[min_c[0]:max_c[0]+1, min_c[1]:max_c[1]+1, :max_c[2]+1] = 1
        for zone in self.env.no_fly_zones:
            min_c = self._world_to_grid((zone[0], zone[1], 0)); max_c = self._world_to_grid((zone[2], zone[3], config.MAX_ALTITUDE))
            grid[min_c[0]:max_c[0]+1, min_c[1]:max_c[1]+1, :] = 1
        return grid
    def _world_to_grid(self, pos):
        x_m = (pos[0] - self.origin_lon) * 111000 * np.cos(np.radians(self.origin_lat)); y_m = (pos[1] - self.origin_lat) * 111000
        grid_pos = (int(x_m / self.resolution), int(y_m / self.resolution), int(pos[2] / self.resolution))
        return tuple(np.clip(grid_pos, 0, np.array(self.grid_shape) - 1))
    def _grid_to_world(self, grid_pos):
        x_m, y_m, z_m = grid_pos[0] * self.resolution, grid_pos[1] * self.resolution, grid_pos[2] * self.resolution
        lon = self.origin_lon + x_m / (111000 * np.cos(np.radians(self.origin_lat))); lat = self.origin_lat + y_m / 111000
        return (lon, lat, z_m)
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
                neighbor = tuple(np.array(curr) + move)
                if not (0 <= neighbor[0] < self.grid_shape[0] and 0 <= neighbor[1] < self.grid_shape[1] and 0 <= neighbor[2] < self.grid_shape[2]): continue
                if neighbor in self.node_map: return neighbor
                if neighbor not in visited: visited.add(neighbor); q.append(neighbor)
        return None

    # --- Offline Heuristic Pre-computation ---
    def _initialize_waypoints(self):
        """Creates a static list of strategic waypoints for the HLUT."""
        waypoints = set()
        cruising_alt_grid = int(config.DEFAULT_CRUISING_ALTITUDE / self.resolution)
        # Add corners of the map
        waypoints.add((5, 5, cruising_alt_grid))
        waypoints.add((self.grid_shape[0]-5, 5, cruising_alt_grid))
        waypoints.add((5, self.grid_shape[1]-5, cruising_alt_grid))
        waypoints.add((self.grid_shape[0]-5, self.grid_shape[1]-5, cruising_alt_grid))
        # Add corners around no-fly zones
        for zone in self.env.no_fly_zones:
            min_c = self._world_to_grid((zone[0], zone[1], 0)); max_c = self._world_to_grid((zone[2], zone[3], 0))
            waypoints.add((min_c[0]-2, min_c[1]-2, cruising_alt_grid)); waypoints.add((max_c[0]+2, max_c[1]+2, cruising_alt_grid))
            waypoints.add((min_c[0]-2, max_c[1]+2, cruising_alt_grid)); waypoints.add((max_c[0]+2, min_c[1]-2, cruising_alt_grid))
        
        valid_waypoints = {self._find_nearest_valid_node(wp) for wp in waypoints}
        return list(filter(None, valid_waypoints))

    def _precompute_heuristic_lookup_table(self):
        """
        OFFLINE STEP: Pre-computes optimal paths between all waypoints in a
        ZERO-WIND environment and stores their cost and geometry.
        """
        logging.info(f"Starting pre-computation of QUBO heuristic table for {len(self.waypoints)} waypoints...")
        start_time = time.time()
        hlut = {wp: {} for wp in self.waypoints}
        # Use a fixed zero payload and balanced weights for a neutral baseline
        baseline_payload, baseline_weights = 0, {'time': 0.5, 'energy': 0.5}

        for wp_a, wp_b in itertools.combinations(self.waypoints, 2):
            if wp_a == wp_b: continue
            cost, path_coords = self._solve_local_qubo(wp_a, wp_b, baseline_payload, baseline_weights, use_zero_wind=True)
            if path_coords:
                world_path = [self._grid_to_world(c) for c in path_coords]
                hlut[wp_a][wp_b] = (cost, world_path)
                hlut[wp_b][wp_a] = (cost, world_path[::-1]) # Store reverse path
        
        end_time = time.time()
        logging.info(f"HLUT pre-computation finished in {end_time - start_time:.2f} seconds.")
        return hlut

    def _solve_local_qubo(self, start_coord, end_coord, payload_kg, weights, use_zero_wind=False):
        """Solves a localized QUBO problem. Can be forced into a zero-wind context."""
        sub_grid_nodes = {coord: self.node_map[coord] for coord in self.node_map if np.all(np.minimum(start_coord, end_coord) - 4 <= coord) and np.all(coord <= np.maximum(start_coord, end_coord) + 4)}
        if not sub_grid_nodes or start_coord not in sub_grid_nodes or end_coord not in sub_grid_nodes:
            path = a_star_search(start_coord, end_coord, self.grid, self.moves)
            if not path: return float('inf'), None
            cost = self._calculate_path_cost([self._grid_to_world(c) for c in path], payload_kg, weights, use_zero_wind)
            return cost, path

        heuristic_path = a_star_search(start_coord, end_coord, self.grid, self.moves, sub_grid_nodes)
        if not heuristic_path: return float('inf'), None
        k = min(len(heuristic_path) + 2, 10) # Cap path length for QUBO efficiency
        
        x = {(i, j): pyqubo.Binary(f'x_{i}_{j}') for i in range(k) for j in sub_grid_nodes.values()}
        cost_h, max_cost = 0, 0
        wind_vector = np.array([0,0,0]) if use_zero_wind else None # Performance optimization

        for i in range(k - 1):
            for u_c, u_idx in sub_grid_nodes.items():
                for move in self.moves:
                    v_c = tuple(np.array(u_c) + move);
                    if v_c in sub_grid_nodes:
                        v_idx = sub_grid_nodes[v_c]; p1, p2 = self._grid_to_world(u_c), self._grid_to_world(v_c)
                        # If wind_vector is not pre-set to zero, fetch it
                        wind = wind_vector if wind_vector is not None else self.env.weather.get_wind_at_location(p1[0], p1[1])
                        t, e = self.predictor.predict(p1, p2, payload_kg, wind)
                        if t != float('inf'): cost = int((t * weights['time'] + e * weights['energy']) * 100); max_cost = max(max_cost, cost); cost_h += cost * x[i, u_idx] * x[i+1, v_idx]
        
        P = max_cost * 1.5 + 100; start_idx, end_idx = self.node_map[start_coord], self.node_map[end_coord]
        one_node_per_layer = sum((sum(x[i, j] for j in sub_grid_nodes.values()) - 1)**2 for i in range(k))
        start_end = (x[0, start_idx] - 1)**2 + (x[k-1, end_idx] - 1)**2
        H = cost_h + P * (one_node_per_layer + start_end)
        
        model = H.compile(); qubo, _ = model.to_qubo(); sampler = SimulatedAnnealingSampler(); sampleset = sampler.sample_qubo(qubo, num_reads=5, num_sweeps=500)
        decoded = model.decode_sample(sampleset.first.sample, vartype='BINARY')
        
        path_indices = sorted([self.reverse_node_map.index(idx) for i in range(k) for idx, val in sub_grid_nodes.items() if decoded.sample.get(f'x_{i}_{val}') == 1])
        path_coords = heuristic_path # Default to heuristic as a fallback
        if path_indices: path_coords = [self.reverse_node_map[idx] for idx in path_indices]

        final_path = [path_coords[0]]; [final_path.append(p) for p in path_coords[1:] if p != final_path[-1]]
        cost = self._calculate_path_cost([self._grid_to_world(c) for c in final_path], payload_kg, weights, use_zero_wind)
        return cost, final_path

    # --- Online Pathfinding with Dynamic Heuristic ---
    def _calculate_path_cost(self, world_path, payload_kg, weights, use_zero_wind=False):
        total_time, total_energy = 0, 0
        for i in range(len(world_path) - 1):
            p1, p2 = world_path[i], world_path[i+1]
            wind = np.array([0,0,0]) if use_zero_wind else self.env.weather.get_wind_at_location(p1[0], p1[1])
            t, e = self.predictor.predict(p1, p2, payload_kg, wind)
            if t == float('inf'): return float('inf')
            total_time += t; total_energy += e
        return weights['time'] * total_time + weights['energy'] * total_energy

    def _calculate_wind_impact_on_path(self, path, payload_kg, weights):
        """FAST, CLASSICAL, REAL-TIME: Calculates cost delta on a path by querying the live weather."""
        cost_with_wind = self._calculate_path_cost(path, payload_kg, weights, use_zero_wind=False)
        cost_no_wind = self._calculate_path_cost(path, payload_kg, weights, use_zero_wind=True)
        return cost_with_wind - cost_no_wind if cost_with_wind != float('inf') else float('inf')

    def _find_closest_waypoint(self, coord):
        """Finds the nearest pre-computed waypoint to a given grid coordinate."""
        return min(self.waypoints, key=lambda wp: np.linalg.norm(np.array(wp) - np.array(coord)))

    def dynamic_quantum_heuristic(self, current_coord, end_coord, payload_kg, weights):
        """The real-time heuristic function used by A*. Corrects pre-computed paths with live wind."""
        wp_a = self._find_closest_waypoint(current_coord); wp_b = self._find_closest_waypoint(end_coord)
        
        # Look up pre-computed baseline cost and path geometry
        if wp_a in self.heuristic_lookup_table and wp_b in self.heuristic_lookup_table[wp_a]:
            baseline_cost, baseline_path = self.heuristic_lookup_table[wp_a][wp_b]
            # Calculate the real-time "delta" cost from wind
            wind_delta_cost = self._calculate_wind_impact_on_path(baseline_path, payload_kg, weights)
            # The final heuristic is the baseline corrected by real-time conditions
            return baseline_cost + wind_delta_cost
        
        # Fallback to simple distance if waypoints not found (should be rare)
        return calculate_distance_3d(self._grid_to_world(current_coord), self._grid_to_world(end_coord))

    def find_path(self, start_pos, end_pos, payload_kg, weights):
        """The main online pathfinding function using A* with the dynamic quantum heuristic."""
        start_coord, end_coord = self._world_to_grid(start_pos), self._world_to_grid(end_pos)
        start_coord, end_coord = self._find_nearest_valid_node(start_coord), self._find_nearest_valid_node(end_coord)
        if not start_coord or not end_coord: return None, "Error: Start or End is in an invalid area."

        logging.info("Starting online A* search with dynamic quantum heuristic...")
        # The heuristic function `h(n)` for A* is now our dynamic_quantum_heuristic
        heuristic_lambda = lambda n, e: self.dynamic_quantum_heuristic(n, e, payload_kg, weights)
        
        path_coords = a_star_search(start_coord, end_coord, self.grid, self.moves, heuristic_func=heuristic_lambda)
        
        if not path_coords: return None, "Error: A* failed to find a valid path"
        
        logging.info("Online A* search completed successfully.")
        return [self._grid_to_world(c) for c in path_coords], "Real-time Optimal (QUBO Guided)"