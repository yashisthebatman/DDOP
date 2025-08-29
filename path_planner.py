# path_planner.py
import numpy as np
import heapq
import config

class PathPlanner3D:
    def __init__(self, env, predictor):
        self.env = env
        self.predictor = predictor
        self.resolution = 100 # meters per grid cell
        self.grid = self._create_grid()
        self._populate_obstacles()

    def _create_grid(self):
        # Convert lat/lon bounds to a local meter grid for pathfinding
        self.origin_lon, self.origin_lat = config.AREA_BOUNDS[0], config.AREA_BOUNDS[1]
        width_m = (config.AREA_BOUNDS[2] - self.origin_lon) * 111000 * np.cos(np.radians(self.origin_lat))
        height_m = (config.AREA_BOUNDS[3] - self.origin_lat) * 111000
        
        x_dim = int(width_m / self.resolution) + 1
        y_dim = int(height_m / self.resolution) + 1
        z_dim = int(config.MAX_ALTITUDE / self.resolution) + 1
        grid = np.zeros((x_dim, y_dim, z_dim), dtype=int)
        min_alt_grid = int(config.MIN_ALTITUDE / self.resolution)
        grid[:, :, :min_alt_grid] = 1
        return grid

    def _world_to_grid(self, pos_lon_lat_alt):
        x_m = (pos_lon_lat_alt[0] - self.origin_lon) * 111000 * np.cos(np.radians(self.origin_lat))
        y_m = (pos_lon_lat_alt[1] - self.origin_lat) * 111000
        z_m = pos_lon_lat_alt[2]
        grid_x = int(x_m / self.resolution)
        grid_y = int(y_m / self.resolution)
        grid_z = int(z_m / self.resolution)
        return tuple(np.clip([grid_x, grid_y, grid_z], 0, np.array(self.grid.shape) - 1))

    def _grid_to_world(self, grid_pos):
        x_m = grid_pos[0] * self.resolution
        y_m = grid_pos[1] * self.resolution
        z_m = grid_pos[2] * self.resolution
        lon = self.origin_lon + x_m / (111000 * np.cos(np.radians(self.origin_lat)))
        lat = self.origin_lat + y_m / 111000
        return (lon, lat, z_m)

    def _populate_obstacles(self):
        """Populates the grid with rectangular building obstacles."""
        for b in self.env.buildings:
            half_len_lon = b.size_xy[0] / 2
            half_wid_lat = b.size_xy[1] / 2
            
            min_corner = self._world_to_grid((b.center_xy[0] - half_len_lon, b.center_xy[1] - half_wid_lat, 0))
            max_corner = self._world_to_grid((b.center_xy[0] + half_len_lon, b.center_xy[1] + half_wid_lat, b.height))
            
            min_x, min_y, _ = min_corner
            max_x, max_y, max_z = max_corner
            
            self.grid[min_x:max_x+1, min_y:max_y+1, :max_z+1] = 1

    def find_path(self, start_pos, end_pos):
        """A* search on the 3D grid."""
        start_node, end_node = self._world_to_grid(start_pos), self._world_to_grid(end_pos)
        open_list = [(0, start_node)]
        came_from, g_score = {}, {start_node: 0}
        
        while open_list:
            _, current_node = heapq.heappop(open_list)
            if current_node == end_node: return self._reconstruct_path(came_from, current_node)
            for move in [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]:
                neighbor_node = tuple(np.array(current_node) + move)
                if not (0 <= neighbor_node[0] < self.grid.shape[0] and 0 <= neighbor_node[1] < self.grid.shape[1] and 0 <= neighbor_node[2] < self.grid.shape[2]) or self.grid[neighbor_node] == 1:
                    continue
                move_cost = np.linalg.norm(np.array(move) * self.resolution)
                tentative_g_score = g_score[current_node] + move_cost
                if tentative_g_score < g_score.get(neighbor_node, float('inf')):
                    came_from[neighbor_node], g_score[neighbor_node] = current_node, tentative_g_score
                    heuristic = np.linalg.norm((np.array(end_node) - np.array(neighbor_node)) * self.resolution)
                    f_score = tentative_g_score + heuristic
                    heapq.heappush(open_list, (f_score, neighbor_node))
        print(f"WARN: A* path not found from {start_pos} to {end_pos}")
        return None

    def _reconstruct_path(self, came_from, current):
        path = [self._grid_to_world(current)]
        while current in came_from:
            current = came_from[current]; path.append(self._grid_to_world(current))
        return path[::-1]