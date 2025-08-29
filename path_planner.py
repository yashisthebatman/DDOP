# path_planner.py
import numpy as np
import heapq

class PathPlanner3D:
    def __init__(self, env, grid_resolution=50):
        self.env = env
        self.resolution = grid_resolution
        self.grid = self._create_grid()
        self._populate_obstacles()

    def _create_grid(self):
        bounds = self.env.bounds
        max_height = 500  # Define a max flight altitude
        x_dim = int((bounds[2] - bounds[0]) / self.resolution) + 1
        y_dim = int((bounds[3] - bounds[1]) / self.resolution) + 1
        z_dim = int(max_height / self.resolution) + 1
        return np.zeros((x_dim, y_dim, z_dim), dtype=int)

    def _populate_obstacles(self):
        for building in self.env.buildings:
            min_x = int((building.center_xy[0] - building.radius) / self.resolution)
            max_x = int((building.center_xy[0] + building.radius) / self.resolution)
            min_y = int((building.center_xy[1] - building.radius) / self.resolution)
            max_y = int((building.center_xy[1] + building.radius) / self.resolution)
            max_z = int(building.height / self.resolution)
            
            self.grid[min_x:max_x, min_y:max_y, :max_z] = 1 # Mark as obstacle

    def _world_to_grid(self, pos):
        return tuple((np.array(pos) / self.resolution).astype(int))

    def _grid_to_world(self, grid_pos):
        return tuple((np.array(grid_pos) * self.resolution) + self.resolution / 2)

    def find_path(self, start_pos, end_pos):
        """A* algorithm to find a path in the 3D grid."""
        start_node = self._world_to_grid(start_pos)
        end_node = self._world_to_grid(end_pos)
        
        open_list = []
        heapq.heappush(open_list, (0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        
        while open_list:
            _, current = heapq.heappop(open_list)
            
            if current == end_node:
                return self._reconstruct_path(came_from, current)
            
            for dx, dy, dz in [(0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)]:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                
                if not (0 <= neighbor[0] < self.grid.shape[0] and \
                        0 <= neighbor[1] < self.grid.shape[1] and \
                        0 <= neighbor[2] < self.grid.shape[2]):
                    continue
                
                if self.grid[neighbor] == 1:
                    continue
                
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + np.linalg.norm(np.array(neighbor) - np.array(end_node))
                    heapq.heappush(open_list, (f_score, neighbor))
        return None

    def _reconstruct_path(self, came_from, current):
        path = [self._grid_to_world(current)]
        while current in came_from:
            current = came_from[current]
            path.append(self._grid_to_world(current))
        return path[::-1] 

    def generate_full_trajectory(self, waypoints):
        """Generates a detailed A* path for a sequence of VRP waypoints."""
        full_path = []
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i+1]
            segment = self.find_path(start, end)
            if segment:
                full_path.extend(segment[:-1]) 
            else:
                print(f"WARN: A* could not find a path from {start} to {end}")
                return None 
        full_path.append(waypoints[-1])
        return full_path