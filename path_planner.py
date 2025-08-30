# path_planner.py
import numpy as np
import pyqubo
from dwave.samplers import SimulatedAnnealingSampler
import logging

import config
from utils.heuristics import a_star_search # Now used ONLY for problem sizing

class PathPlanner3D:
    def __init__(self, env, predictor):
        self.env = env
        self.predictor = predictor
        self.resolution = 125 # Slightly increased resolution for the new model
        self.grid_shape, self.origin_lon, self.origin_lat = self._get_grid_params()
        self.grid = self._create_and_populate_grid()
        self.node_map, self.reverse_node_map = self._create_node_maps()
        self.moves = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1] if not (dx == 0 and dy == 0 and dz == 0)]
        logging.info(f"PathPlanner initialized with grid shape {self.grid_shape}. Total valid nodes: {len(self.node_map)}")

    def _get_grid_params(self):
        origin_lon, origin_lat = config.AREA_BOUNDS[0], config.AREA_BOUNDS[1]
        width_m = (config.AREA_BOUNDS[2] - origin_lon) * 111000 * np.cos(np.radians(origin_lat))
        height_m = (config.AREA_BOUNDS[3] - origin_lat) * 111000
        x_dim, y_dim = int(width_m / self.resolution) + 1, int(height_m / self.resolution) + 1
        z_dim = int(config.MAX_ALTITUDE / self.resolution) + 1
        return (x_dim, y_dim, z_dim), origin_lon, origin_lat

    def _create_and_populate_grid(self):
        grid = np.zeros(self.grid_shape, dtype=int)
        min_alt_grid = int(config.MIN_ALTITUDE / self.resolution)
        grid[:, :, :min_alt_grid] = 1 # Enforce hard minimum altitude
        for b in self.env.buildings:
            min_c = self._world_to_grid((b.center_xy[0] - b.size_xy[0]/2, b.center_xy[1] - b.size_xy[1]/2, 0))
            max_c = self._world_to_grid((b.center_xy[0] + b.size_xy[0]/2, b.center_xy[1] + b.size_xy[1]/2, b.height))
            grid[min_c[0]:max_c[0]+1, min_c[1]:max_c[1]+1, :max_c[2]+1] = 1
        for zone in self.env.no_fly_zones:
            min_c = self._world_to_grid((zone[0], zone[1], 0))
            max_c = self._world_to_grid((zone[2], zone[3], config.MAX_ALTITUDE))
            grid[min_c[0]:max_c[0]+1, min_c[1]:max_c[1]+1, :] = 1
        return grid

    def _world_to_grid(self, pos):
        x_m = (pos[0] - self.origin_lon) * 111000 * np.cos(np.radians(self.origin_lat))
        y_m = (pos[1] - self.origin_lat) * 111000
        grid_pos = (int(x_m / self.resolution), int(y_m / self.resolution), int(pos[2] / self.resolution))
        return tuple(np.clip(grid_pos, 0, np.array(self.grid_shape) - 1))

    def _grid_to_world(self, grid_pos):
        x_m, y_m, z_m = grid_pos[0] * self.resolution, grid_pos[1] * self.resolution, grid_pos[2] * self.resolution
        lon = self.origin_lon + x_m / (111000 * np.cos(np.radians(self.origin_lat)))
        lat = self.origin_lat + y_m / 111000
        return (lon, lat, z_m)

    def _create_node_maps(self):
        node_map, reverse_node_map = {}, []
        idx = 0
        for x in range(self.grid_shape[0]):
            for y in range(self.grid_shape[1]):
                for z in range(self.grid_shape[2]):
                    if self.grid[x, y, z] == 0:
                        coord = (x, y, z); node_map[coord] = idx; reverse_node_map.append(coord); idx += 1
        return node_map, reverse_node_map

    def find_path(self, start_pos, end_pos, payload_kg, weights):
        """
        Finds the optimal path using a pure, layered-graph QUBO formulation.
        A* is used ONLY to determine the optimal number of layers, k.
        """
        start_coord, end_coord = self._world_to_grid(start_pos), self._world_to_grid(end_pos)

        if start_coord not in self.node_map or end_coord not in self.node_map:
            logging.error("Start or end coordinate is inside an obstacle.")
            return None, "Error: Invalid Coordinates"

        # --- Adaptive Problem Sizing ---
        logging.info("Sizing problem: Using A* to find optimal path length (k)...")
        heuristic_path = a_star_search(start_coord, end_coord, self.grid, self.moves)
        if not heuristic_path:
            logging.error("Feasibility check FAILED. No path exists between start and end.")
            return None, "Error: Route is physically impossible"
        
        # Set k to the length of the heuristic path + a small buffer for optimization freedom
        k = len(heuristic_path) + 2
        logging.info(f"Optimal path length determined. Setting up Pure QUBO with k={k} layers.")

        # --- QUBO Formulation ---
        start_node_idx, end_node_idx = self.node_map[start_coord], self.node_map[end_coord]

        # x_i_j = 1 if drone is at node j at step i
        x = {(i, j): pyqubo.Binary(f'x_{i}_{j}') for i in range(k) for j in self.node_map.values()}

        # -- Cost Hamiltonian (H_cost) --
        cost_hamiltonian, max_cost = 0, 0
        for i in range(k - 1): # For each layer transition
            for u_coord, u_idx in self.node_map.items():
                for move in self.moves: # For each possible connection
                    v_coord = tuple(np.array(u_coord) + move)
                    if v_coord in self.node_map:
                        v_idx = self.node_map[v_coord]
                        p1, p2 = self._grid_to_world(u_coord), self._grid_to_world(v_coord)
                        wind = self.env.weather.get_wind_at_location(p1[0], p1[1])
                        t, e = self.predictor.predict(p1, p2, payload_kg, wind) # Ignoring turning cost in this model
                        if t != float('inf'):
                            cost = int((t * weights['time'] + e * weights['energy']) * 100)
                            max_cost = max(max_cost, cost)
                            # Add term: cost * x_i_u * x_{i+1}_v
                            cost_hamiltonian += cost * x[i, u_idx] * x[i+1, v_idx]

        # -- Constraint Hamiltonian (H_constraint) --
        P = max_cost * 1.5 + 100 # Dynamic penalty
        
        # C1: Drone must be in exactly one place at each step
        one_node_per_layer = sum((sum(x[i, j] for j in self.node_map.values()) - 1)**2 for i in range(k))
        
        # C2: Force start and end points
        start_end_constraint = (x[0, start_node_idx] - 1)**2 + (x[k-1, end_node_idx] - 1)**2

        H = cost_hamiltonian + P * (one_node_per_layer + start_end_constraint)

        # --- Solve and Decode ---
        logging.info("QUBO model compiled. Sampling with high-effort D-Wave annealer...")
        model = H.compile()
        qubo, _ = model.to_qubo()
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample_qubo(qubo, num_reads=10, num_sweeps=2000)
        decoded_sample = model.decode_sample(sampleset.first.sample, vartype='BINARY')

        path_indices = []
        for i in range(k):
            found_node_in_layer = False
            for j in self.node_map.values():
                if decoded_sample.sample.get(f'x_{i}_{j}') == 1:
                    path_indices.append(j)
                    found_node_in_layer = True
                    break
            if not found_node_in_layer:
                logging.error(f"Pure QUBO solver failed: Layer {i} has no active node.")
                return None, "Error: Solver failed to produce a continuous path"

        if not path_indices or path_indices[0] != start_node_idx or path_indices[-1] != end_node_idx:
            logging.error("Pure QUBO solver failed: Path does not start or end correctly.")
            return None, "Error: Solver failed to meet constraints"

        # Remove consecutive duplicates from the path
        final_path_indices = [path_indices[0]]
        for i in range(1, len(path_indices)):
            if path_indices[i] != path_indices[i-1]:
                final_path_indices.append(path_indices[i])
        
        logging.info("Pure QUBO path successfully found and decoded.")
        return [self._grid_to_world(self.reverse_node_map[idx]) for idx in final_path_indices], "Pure QUBO Optimal"