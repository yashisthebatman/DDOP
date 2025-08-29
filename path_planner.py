# path_planner.py
import numpy as np
from ortools.graph.python import min_cost_flow
import pyqubo
from dwave.samplers import SimulatedAnnealingSampler
import logging

import config

class PathPlanner3D:
    def __init__(self, env, predictor):
        self.env = env
        self.predictor = predictor
        self.resolution = 150
        self.grid_shape, self.origin_lon, self.origin_lat = self._get_grid_params()
        self.grid = self._create_and_populate_grid()
        self.node_map, self.reverse_node_map = self._create_node_maps()
        self.smcf = self._build_or_tools_graph()
        logging.info(f"PathPlanner initialized with grid shape {self.grid_shape}. Total valid nodes: {len(self.node_map)}")

    # --- Grid and Node Setup ---
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

    # --- Main Public Method ---
    def find_path(self, start_pos, end_pos, payload_kg, solver_choice):
        if "OR-Tools" in solver_choice:
            return self.find_path_with_or_tools(start_pos, end_pos, payload_kg)
        else:
            return self.find_path_with_qubo(start_pos, end_pos, payload_kg)

    # --- Google OR-Tools Implementation ---
    def _build_or_tools_graph(self):
        smcf = min_cost_flow.SimpleMinCostFlow()
        moves = [(dx, dy, dz) for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1] if not (dx==0 and dy==0 and dz==0)]
        for start_coord, start_node_idx in self.node_map.items():
            for move in moves:
                end_coord = tuple(np.array(start_coord) + move)
                if end_coord in self.node_map:
                    end_node_idx = self.node_map[end_coord]
                    smcf.add_arc_with_capacity_and_unit_cost(start_node_idx, end_node_idx, 1, 0)
        return smcf

    def find_path_with_or_tools(self, start_pos, end_pos, payload_kg):
        start_coord, end_coord = self._world_to_grid(start_pos), self._world_to_grid(end_pos)
        if start_coord not in self.node_map or end_coord not in self.node_map:
            logging.error("OR-Tools: Start/end node in obstacle.")
            return None
        start_node, end_node = self.node_map[start_coord], self.node_map[end_coord]

        infinite_cost = 9999999
        for i in range(self.smcf.num_arcs()):
            p1_world = self._grid_to_world(self.reverse_node_map[self.smcf.tail(i)])
            p2_world = self._grid_to_world(self.reverse_node_map[self.smcf.head(i)])
            wind = self.env.weather.get_wind_at_location(p1_world[0], p1_world[1])
            time, energy = self.predictor.predict(p1_world, p2_world, payload_kg, wind)
            
            if time == float('inf') or energy == float('inf'):
                cost = infinite_cost
            else:
                cost = int((time * 0.5 + energy * 0.5) * 100)
            self.smcf.set_arc_unit_cost(i, cost)

        self.smcf.set_node_supply(start_node, 1)
        self.smcf.set_node_supply(end_node, -1)
        status = self.smcf.solve()
        self.smcf.set_node_supply(start_node, 0)
        self.smcf.set_node_supply(end_node, 0)

        if status != self.smcf.OPTIMAL:
            logging.warning(f"OR-Tools pathfinder failed with status {status}.")
            return None
        
        path = [self._grid_to_world(start_coord)]
        curr = start_node
        while curr != end_node:
            found = False
            for i in range(self.smcf.num_arcs()):
                if self.smcf.tail(i) == curr and self.smcf.flow(i) > 0:
                    curr = self.smcf.head(i)
                    path.append(self._grid_to_world(self.reverse_node_map[curr]))
                    found = True
                    break
            if not found: return None
        return path

    # --- QUBO Implementation ---
    def find_path_with_qubo(self, start_pos, end_pos, payload_kg):
        start_coord, end_coord = self._world_to_grid(start_pos), self._world_to_grid(end_pos)
        if start_coord not in self.node_map or end_coord not in self.node_map:
            logging.error("QUBO: Start/end node in obstacle.")
            return None
        start_node, end_node = self.node_map[start_coord], self.node_map[end_coord]

        x = {}
        moves = [(dx, dy, dz) for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1] if not (dx==0 and dy==0 and dz==0)]
        for u_coord, u_idx in self.node_map.items():
            for move in moves:
                v_coord = tuple(np.array(u_coord) + move)
                if v_coord in self.node_map:
                    v_idx = self.node_map[v_coord]
                    x[(u_idx, v_idx)] = pyqubo.Binary(f'x_{u_idx}_{v_idx}')

        cost_hamiltonian = 0
        for (u_idx, v_idx), var in x.items():
            p1_world = self._grid_to_world(self.reverse_node_map[u_idx])
            p2_world = self._grid_to_world(self.reverse_node_map[v_idx])
            wind = self.env.weather.get_wind_at_location(p1_world[0], p1_world[1])
            time, energy = self.predictor.predict(p1_world, p2_world, payload_kg, wind)
            cost = int((time * 0.5 + energy * 0.5) * 100)
            cost_hamiltonian += cost * var
        
        # Increased penalty to better enforce constraints
        P1 = 10000 
        start_constraint = (sum(x.get((start_node, v_idx), 0) for v_idx in self.node_map.values()) - 1)**2
        end_constraint = (sum(x.get((u_idx, end_node), 0) for u_idx in self.node_map.values()) - 1)**2
        intermediate_constraint = 0
        for i_idx in self.node_map.values():
            if i_idx != start_node and i_idx != end_node:
                inflow = sum(x.get((u_idx, i_idx), 0) for u_idx in self.node_map.values())
                outflow = sum(x.get((i_idx, v_idx), 0) for v_idx in self.node_map.values())
                intermediate_constraint += (inflow - outflow)**2

        H = cost_hamiltonian + P1 * (start_constraint + end_constraint + intermediate_constraint)
        model = H.compile()
        qubo, offset = model.to_qubo()
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample_qubo(qubo, num_reads=15, num_sweeps=1000)
        decoded_sample = model.decode_sample(sampleset.first.sample, vartype='BINARY')
        
        # --- Robust Path Reconstruction ---
        path = [self._grid_to_world(start_coord)]
        curr_node = start_node
        
        # Create a lookup dictionary from the solver's result for efficiency
        solution_arcs = {u: v for (u, v), var in x.items() if decoded_sample.array[var.label][0] > 0.5}

        for _ in range(len(self.node_map)): # Failsafe to prevent infinite loops
            if curr_node not in solution_arcs:
                logging.warning("QUBO path reconstruction failed: Path broken.")
                return None
            
            next_node = solution_arcs[curr_node]
            path.append(self._grid_to_world(self.reverse_node_map[next_node]))
            curr_node = next_node
            
            if curr_node == end_node:
                return path # Success
                
        logging.warning("QUBO path reconstruction failed: Path did not reach destination.")
        return None