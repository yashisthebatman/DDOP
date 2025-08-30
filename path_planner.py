# path_planner.py
import numpy as np
import pyqubo
from dwave.samplers import SimulatedAnnealingSampler
import logging
import heapq

import config
from utils.heuristics import a_star_search

class PathPlanner3D:
    def __init__(self, env, predictor):
        self.env = env
        self.predictor = predictor
        self.resolution = 125
        self.grid_shape, self.origin_lon, self.origin_lat = self._get_grid_params()
        self.grid = self._create_and_populate_grid()
        self.node_map, self.reverse_node_map = self._create_node_maps()
        self.moves = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1] if not (dx == 0 and dy == 0 and dz == 0)]
        logging.info("Hierarchical PathPlanner initialized.")

    # --- Grid and Node Setup (Verified: Correct) ---
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
        
    def _create_waypoint_graph(self, start_coord, end_coord):
        waypoints = {start_coord, end_coord}
        cruising_alt_grid = int(config.DEFAULT_CRUISING_ALTITUDE / self.resolution)
        for zone in self.env.no_fly_zones:
            min_c = self._world_to_grid((zone[0], zone[1], 0)); max_c = self._world_to_grid((zone[2], zone[3], 0))
            waypoints.add((min_c[0], min_c[1], cruising_alt_grid)); waypoints.add((max_c[0], max_c[1], cruising_alt_grid))
            waypoints.add((min_c[0], max_c[1], cruising_alt_grid)); waypoints.add((max_c[0], min_c[1], cruising_alt_grid))
        valid_waypoints = {self._find_nearest_valid_node(wp) for wp in waypoints}
        return list(filter(None, valid_waypoints))

    def _solve_local_qubo(self, start_coord, end_coord, payload_kg, weights):
        min_coords = np.minimum(start_coord, end_coord) - 5; max_coords = np.maximum(start_coord, end_coord) + 5
        sub_grid_nodes = {coord: idx for coord, idx in self.node_map.items() if np.all(min_coords <= coord) and np.all(coord <= max_coords)}
        if not sub_grid_nodes or start_coord not in sub_grid_nodes or end_coord not in sub_grid_nodes:
            path = a_star_search(start_coord, end_coord, self.grid, self.moves);
            if not path: return float('inf'), None
            cost = self._calculate_path_cost([self._grid_to_world(c) for c in path], payload_kg, weights); return cost, path
        
        heuristic_path = a_star_search(start_coord, end_coord, self.grid, self.moves, sub_grid_nodes)
        if not heuristic_path: return float('inf'), None
        k = len(heuristic_path) + 2
        if k <= 2:
             path = [start_coord, end_coord]; cost = self._calculate_path_cost([self._grid_to_world(c) for c in path], payload_kg, weights); return cost, path

        x = {(i, j): pyqubo.Binary(f'x_{i}_{j}') for i in range(k) for j in sub_grid_nodes.values()}
        cost_h, max_cost = 0, 0
        for i in range(k - 1):
            for u_c, u_idx in sub_grid_nodes.items():
                for move in self.moves:
                    v_c = tuple(np.array(u_c) + move)
                    if v_c in sub_grid_nodes:
                        v_idx = sub_grid_nodes[v_c]; p1, p2 = self._grid_to_world(u_c), self._grid_to_world(v_c)
                        wind = self.env.weather.get_wind_at_location(p1[0], p1[1]); t, e = self.predictor.predict(p1, p2, payload_kg, wind)
                        if t != float('inf'): cost = int((t * weights['time'] + e * weights['energy']) * 100); max_cost = max(max_cost, cost); cost_h += cost * x[i, u_idx] * x[i+1, v_idx]
        
        P = max_cost * 1.5 + 100; start_idx, end_idx = self.node_map[start_coord], self.node_map[end_coord]
        one_node_per_layer = sum((sum(x[i, j] for j in sub_grid_nodes.values()) - 1)**2 for i in range(k)); start_end = (x[0, start_idx] - 1)**2 + (x[k-1, end_idx] - 1)**2
        H = cost_h + P * (one_node_per_layer + start_end)
        model = H.compile(); qubo, _ = model.to_qubo(); sampler = SimulatedAnnealingSampler(); sampleset = sampler.sample_qubo(qubo, num_reads=5, num_sweeps=500)
        decoded = model.decode_sample(sampleset.first.sample, vartype='BINARY')
        
        # *** CRITICAL BUG FIX ***: Collect integer indices (val), not coordinate tuples (idx).
        raw_path_indices = [val for i in range(k) for idx, val in sub_grid_nodes.items() if decoded.sample.get(f'x_{i}_{val}') == 1]
        
        path_coords = heuristic_path # Default to heuristic as a robust fallback
        if raw_path_indices and raw_path_indices[0] == start_idx and raw_path_indices[-1] == end_idx:
            path_coords = [self.reverse_node_map[idx] for idx in raw_path_indices]

        final_path_coords = [path_coords[0]]; [final_path_coords.append(p) for p in path_coords[1:] if p != final_path_coords[-1]]
        cost = self._calculate_path_cost([self._grid_to_world(c) for c in final_path_coords], payload_kg, weights)
        return cost, final_path_coords

    def _calculate_path_cost(self, world_path, payload_kg, weights):
        total_time, total_energy = 0, 0
        for i in range(len(world_path) - 1):
            p1, p2 = world_path[i], world_path[i+1]
            wind = self.env.weather.get_wind_at_location(p1[0], p1[1]); t, e = self.predictor.predict(p1, p2, payload_kg, wind)
            total_time += t; total_energy += e
        return weights['time'] * total_time + weights['energy'] * total_energy

    def find_path(self, start_pos, end_pos, payload_kg, weights):
        start_coord, end_coord = self._world_to_grid(start_pos), self._world_to_grid(end_pos)
        if start_coord not in self.node_map or end_coord not in self.node_map: return None, "Error: Invalid Coordinates"
        
        manhattan_dist = abs(start_coord[0] - end_coord[0]) + abs(start_coord[1] - end_coord[1])
        if manhattan_dist < 20:
            logging.info("Proximity check PASSED. Solving with a single, direct QUBO.")
            _, path_coords = self._solve_local_qubo(start_coord, end_coord, payload_kg, weights)
            if path_coords is None: return None, "Error: Direct pathfinding failed"
            return [self._grid_to_world(c) for c in path_coords], "Direct QUBO Optimal"

        logging.info("Proximity check FAILED. Using full hierarchical planner.")
        waypoints = self._create_waypoint_graph(start_coord, end_coord)
        
        # --- RE-ARCHITECTED: Performant A* with path caching ---
        open_set = [(0, start_coord)] # (f_score, waypoint)
        came_from = {} # Stores {waypoint: (parent_waypoint, path_segment_to_get_here)}
        g_scores = {wp: float('inf') for wp in waypoints}; g_scores[start_coord] = 0
        
        while open_set:
            _, current_wp = heapq.heappop(open_set)

            if current_wp == end_coord:
                logging.info("Strategic waypoint path found. Stitching final QUBO path...")
                # Reconstruct path by backtracking and collecting cached path segments
                full_path_coords = []
                curr = end_coord
                while curr in came_from:
                    prev, segment = came_from[curr]
                    full_path_coords = segment[:-1] + full_path_coords
                    curr = prev
                full_path_coords = [start_coord] + full_path_coords
                return [self._grid_to_world(c) for c in full_path_coords], "Hierarchical QUBO Optimal"

            for neighbor_wp in waypoints:
                if neighbor_wp == current_wp: continue
                cost, path_segment = self._solve_local_qubo(current_wp, neighbor_wp, payload_kg, weights)
                if cost == float('inf'): continue
                
                tentative_g_score = g_scores[current_wp] + cost
                if tentative_g_score < g_scores.get(neighbor_wp, float('inf')):
                    came_from[neighbor_wp] = (current_wp, path_segment)
                    g_scores[neighbor_wp] = tentative_g_score
                    heuristic = np.linalg.norm(np.array(neighbor_wp) - np.array(end_coord))
                    f_score = tentative_g_score + heuristic
                    heapq.heappush(open_set, (f_score, neighbor_wp))
        
        logging.error("High-level A* failed to find a route between waypoints.")
        return None, "Error: No strategic path found"