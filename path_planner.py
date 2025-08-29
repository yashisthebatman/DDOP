# path_planner.py
import numpy as np
import heapq

import config

class PathPlanner3D:
    def __init__(self, env, predictor, optimization_weights):
        self.env = env
        self.predictor = predictor
        self.weights = optimization_weights
        self.resolution = 50  # Grid resolution in meters
        self.grid = self._create_grid()
        self._populate_obstacles()

    def _create_grid(self):
        bounds = self.env.bounds
        x_dim = int((bounds[2] - bounds[0]) / self.resolution) + 1
        y_dim = int((bounds[3] - bounds[1]) / self.resolution) + 1
        z_dim = int(config.MAX_ALTITUDE / self.resolution) + 1
        grid = np.zeros((x_dim, y_dim, z_dim), dtype=int)
        
        # Enforce min/max altitude by marking out-of-bounds cells as obstacles
        min_alt_grid = int(config.MIN_ALTITUDE / self.resolution)
        grid[:, :, :min_alt_grid] = 1 # Below min altitude
        grid[:, :, int(config.MAX_ALTITUDE / self.resolution):] = 1 # Above max altitude
        return grid

    def _populate_obstacles(self):
        for building in self.env.buildings:
            min_x = int(max(0, (building.center_xy[0] - building.radius)) / self.resolution)
            max_x = int(min(self.grid.shape[0]-1, (building.center_xy[0] + building.radius)) / self.resolution)
            min_y = int(max(0, (building.center_xy[1] - building.radius)) / self.resolution)
            max_y = int(min(self.grid.shape[1]-1, (building.center_xy[1] + building.radius)) / self.resolution)
            max_z = int(building.height / self.resolution)
            self.grid[min_x:max_x, min_y:max_y, :max_z] = 1

    def _world_to_grid(self, pos):
        return tuple(np.clip((np.array(pos) / self.resolution), 0, np.array(self.grid.shape)-1).astype(int))

    def _grid_to_world(self, grid_pos):
        return tuple((np.array(grid_pos) * self.resolution) + self.resolution / 2)

    def find_path(self, start_pos, end_pos):
        """Cost-aware A* algorithm using the ML predictor."""
        start_node, end_node = self._world_to_grid(start_pos), self._world_to_grid(end_pos)
        open_list = [(0, start_node)]
        came_from, g_score = {}, {start_node: 0}
        
        while open_list:
            _, current_node = heapq.heappop(open_list)
            
            if current_node == end_node:
                return self._reconstruct_path(came_from, current_node)
            
            for move in [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]:
                neighbor_node = tuple(np.array(current_node) + move)
                if not (0 <= neighbor_node[0] < self.grid.shape[0] and 0 <= neighbor_node[1] < self.grid.shape[1] and 0 <= neighbor_node[2] < self.grid.shape[2]):
                    continue
                if self.grid[neighbor_node] == 1:
                    continue
                
                # Calculate true cost using ML predictor
                p1 = self._grid_to_world(current_node)
                p2 = self._grid_to_world(neighbor_node)
                time_cost, energy_cost = self.predictor.predict(p1, p2, 2.5, self.env.wind_vector) # Avg payload
                move_cost = self.weights['time'] * time_cost + self.weights['energy'] * energy_cost
                
                tentative_g_score = g_score[current_node] + move_cost
                if tentative_g_score < g_score.get(neighbor_node, float('inf')):
                    came_from[neighbor_node], g_score[neighbor_node] = current_node, tentative_g_score
                    heuristic = np.linalg.norm(np.array(end_pos) - np.array(p2)) / config.DRONE_SPEED_MPS
                    f_score = tentative_g_score + heuristic
                    heapq.heappush(open_list, (f_score, neighbor_node))
        return None

    def _reconstruct_path(self, came_from, current):
        path = [self._grid_to_world(current)]
        while current in came_from:
            current = came_from[current]; path.append(self._grid_to_world(current))
        return path[::-1]

    def generate_full_trajectory(self, waypoints):
        full_path = []
        for i in range(len(waypoints) - 1):
            segment = self.find_path(waypoints[i], waypoints[i+1])
            if segment: full_path.extend(segment[:-1])
            else: return None
        full_path.append(waypoints[-1])
        return full_path

    def process_routes(self, routes, locations_dict, is_qubo=False):
        """Processes routes from a solver, generates trajectories, and calculates costs."""
        final_paths = {}
        if not routes: return {}
        for drone_id, route_data in routes.items():
            readable_route = route_data['readable'] if not is_qubo else route_data
            
            if len(readable_route) <= 1 or (len(readable_route) == 2 and readable_route[0] == readable_route[1]):
                continue # Skip idle drones

            waypoints = [locations_dict[name] for name in readable_route]
            trajectory = self.generate_full_trajectory(waypoints)
            if trajectory:
                total_time, total_energy = 0, 0
                for i in range(len(trajectory) - 1):
                    t, e = self.predictor.predict(trajectory[i], trajectory[i+1], 2.5, self.env.wind_vector)
                    total_time += t; total_energy += e
                
                final_paths[drone_id] = {
                    "waypoints": readable_route, "trajectory": trajectory,
                    "time": total_time, "energy": total_energy
                }
        return final_paths