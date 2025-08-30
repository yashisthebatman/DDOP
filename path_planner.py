# path_planner.py
import numpy as np
import pyqubo
from dwave.samplers import SimulatedAnnealingSampler
import logging

import config
from utils.heuristics import a_star_search

class PathPlanner3D:
    # --- (__init__ and grid methods are unchanged) ---
    def __init__(self, env, predictor):
        self.env = env
        self.predictor = predictor
        self.resolution = 100
        self.grid_shape, self.origin_lon, self.origin_lat = self._get_grid_params()
        self.grid = self._create_and_populate_grid()
        self.node_map, self.reverse_node_map = self._create_node_maps()
        self.moves = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1] if not (dx == 0 and dy == 0 and dz == 0)]
        logging.info(f"PathPlanner initialized with grid shape {self.grid_shape}. Total valid nodes: {len(self.node_map)}")

    def _get_grid_params(self):
        origin_lon, origin_lat = config.AREA_BOUNDS[0], config.AREA_BOUNDS[1]
        width_m = (config.AREA_BOUNDS[2] - origin_lon) * 111000 * np.cos(np.radians(origin_lat))
        height_m = (config.AREA_BOUNDS[3] - origin_lat) * 111000
        x_dim = int(width_m / self.resolution) + 1
        y_dim = int(height_m / self.resolution) + 1
        z_dim = int(config.MAX_ALTITUDE / self.resolution) + 1
        return (x_dim, y_dim, z_dim), origin_lon, origin_lat

    def _create_and_populate_grid(self):
        grid = np.zeros(self.grid_shape, dtype=int)
        min_alt_grid = int(config.MIN_ALTITUDE / self.resolution)
        grid[:, :, :min_alt_grid] = 1
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
        x_m = grid_pos[0] * self.resolution
        y_m = grid_pos[1] * self.resolution
        z_m = grid_pos[2] * self.resolution
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
                        coord = (x, y, z)
                        node_map[coord] = idx
                        reverse_node_map.append(coord)
                        idx += 1
        return node_map, reverse_node_map

    def find_path(self, start_pos, end_pos, payload_kg, weights):
        start_coord = self._world_to_grid(start_pos)
        end_coord = self._world_to_grid(end_pos)

        if start_coord not in self.node_map or end_coord not in self.node_map:
            logging.error("Start or end coordinate is inside an obstacle.")
            return None

        # *** CRITICAL BUG FIX ***
        # The variables were swapped here. It should be end_coord.
        start_node, end_node = self.node_map[start_coord], self.node_map[end_coord]

        heuristic_path_coords = a_star_search(start_coord, end_coord, self.grid, self.moves)
        if not heuristic_path_coords:
            logging.warning("A* pre-search failed to find a path.")
            return None

        corridor_nodes = set(heuristic_path_coords)
        for coord in heuristic_path_coords:
            for move in self.moves:
                neighbor = tuple(np.array(coord) + move)
                if neighbor in self.node_map: corridor_nodes.add(neighbor)

        x = {f'x_{self.node_map[u]}_{self.node_map[v]}': pyqubo.Binary(f'x_{self.node_map[u]}_{self.node_map[v]}')
             for u in corridor_nodes for move in self.moves if (v := tuple(np.array(u) + move)) in corridor_nodes}
        if not x: return [self._grid_to_world(c) for c in heuristic_path_coords]

        max_cost = 0
        cost_terms = []
        for label, var in x.items():
            u_idx, v_idx = map(int, label.split('_')[1:])
            p1 = self._grid_to_world(self.reverse_node_map[u_idx])
            p2 = self._grid_to_world(self.reverse_node_map[v_idx])
            
            # Note: Turning cost is not included in the QUBO formulation itself as it requires
            # knowing the previous point, which complicates the model. It's applied during final
            # cost calculation. The path is still optimized for distance, altitude, and wind.
            wind = self.env.weather.get_wind_at_location(p1[0], p1[1])
            t, e = self.predictor.predict(p1, p2, payload_kg, wind) # Base cost
            if t != float('inf'):
                cost = int((t * weights['time'] + e * weights['energy']) * 100)
                cost_terms.append(cost * var)
                if cost > max_cost: max_cost = cost
        cost_hamiltonian = sum(cost_terms)
        
        P1 = max_cost * 1.5 + 100
        
        start_constraint = (sum(x.get(f'x_{start_node}_{self.node_map[v]}', 0) for v in corridor_nodes if f'x_{start_node}_{self.node_map[v]}' in x) - 1)**2
        end_constraint = (sum(x.get(f'x_{self.node_map[u]}_{end_node}', 0) for u in corridor_nodes if f'x_{self.node_map[u]}_{end_node}' in x) - 1)**2
        intermediate_constraint = sum( (sum(x.get(f'x_{self.node_map[u]}_{i_idx}', 0) for u in corridor_nodes if f'x_{self.node_map[u]}_{i_idx}' in x) -
                                       sum(x.get(f'x_{i_idx}_{self.node_map[v]}', 0) for v in corridor_nodes if f'x_{i_idx}_{self.node_map[v]}' in x))**2
                                       for i_coord, i_idx in self.node_map.items() if i_coord in corridor_nodes and i_idx != start_node and i_idx != end_node )

        H = cost_hamiltonian + P1 * (start_constraint + end_constraint + intermediate_constraint)
        model = H.compile()
        qubo, _ = model.to_qubo()

        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample_qubo(qubo, num_reads=10, num_sweeps=1000)
        decoded_sample = model.decode_sample(sampleset.first.sample, vartype='BINARY')

        path = [self._grid_to_world(start_coord)]
        curr_node = start_node
        solution_arcs = {int(k.split('_')[1]): int(k.split('_')[2]) for k, v in decoded_sample.sample.items() if v > 0.5 and k.startswith('x_')}

        for _ in range(len(corridor_nodes) + 1):
            if curr_node == end_node: break
            if curr_node not in solution_arcs:
                logging.warning("QUBO path reconstruction failed. Falling back to A* path.")
                return [self._grid_to_world(c) for c in heuristic_path_coords]
            next_node = solution_arcs[curr_node]
            path.append(self._grid_to_world(self.reverse_node_map[next_node]))
            curr_node = next_node

        if curr_node != end_node:
            logging.warning("QUBO path did not reach destination. Falling back to A* path.")
            return [self._grid_to_world(c) for c in heuristic_path_coords]
        return path