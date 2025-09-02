# ==============================================================================
# File: utils/jump_point_search.py
# ==============================================================================
import heapq
from itertools import product
import numpy as np

from config import MAX_PATH_LENGTH

class JumpPointSearch:
    def __init__(self, start, goal, is_obstructed_func, heuristic, grid_shape=None):
        self.start = start
        self.goal = goal
        self.is_obstructed = is_obstructed_func
        self.heuristic = heuristic
        self.grid_shape = grid_shape or (1000, 1000, 100)

        self.open_set = []
        self.open_set_map = {}
        self.came_from = {}
        self.g_score = {start: 0}
        
        self.directions = list(product([-1, 0, 1], repeat=3))
        self.directions.remove((0, 0, 0))
        
        self.MAX_JUMP_DEPTH = MAX_PATH_LENGTH
        self.max_iterations = 20000
        
        # FIX: Manage visited set as an instance variable to prevent recursion bugs
        self.jump_visited = set()

    def search(self):
        if self.start == self.goal:
            return [self.start]
            
        h_start = self.heuristic.calculate(self.start)
        heapq.heappush(self.open_set, (h_start, self.start))
        self.open_set_map[self.start] = h_start

        iterations = 0
        while self.open_set and iterations < self.max_iterations:
            iterations += 1
            _, current = heapq.heappop(self.open_set)
            
            if current == self.goal:
                return self._reconstruct_path(current)

            if current in self.open_set_map:
                del self.open_set_map[current]
            
            successors = self._identify_successors(current)
            for successor in successors:
                new_g_score = self.g_score[current] + np.linalg.norm(np.array(current) - np.array(successor))
                
                if successor not in self.g_score or new_g_score < self.g_score.get(successor, float('inf')):
                    self.g_score[successor] = new_g_score
                    f_score = new_g_score + self.heuristic.calculate(successor)
                    
                    if successor not in self.open_set_map:
                        heapq.heappush(self.open_set, (f_score, successor))
                        self.open_set_map[successor] = f_score
                    
                    self.came_from[successor] = current
        
        return None

    def _identify_successors(self, node):
        successors = set()
        parent = self.came_from.get(node)
        
        if parent is None:
            pruned_directions = self.directions
        else:
            direction = tuple(np.sign(np.array(node) - np.array(parent)).astype(int))
            pruned_directions = self._prune_directions(direction)
        
        for d in pruned_directions:
            # FIX: Clear the visited set for each new jump exploration from the current node
            self.jump_visited.clear()
            jump_point = self._jump(node, d, 0)
            if jump_point:
                successors.add(jump_point)
        
        return list(successors)

    # FIX: Removed 'visited' from arguments, uses self.jump_visited instead
    def _jump(self, node, direction, depth):
        if depth > self.MAX_JUMP_DEPTH or node in self.jump_visited:
            return None

        self.jump_visited.add(node)
        
        next_node = (node[0] + direction[0], node[1] + direction[1], node[2] + direction[2])
        
        if not self._is_valid_position(next_node) or self.is_obstructed(next_node):
            return None
        if next_node == self.goal:
            return next_node
        if self._has_forced_neighbor(next_node, direction):
            return next_node

        dx, dy, dz = direction
        # Check for diagonal movement
        if dx != 0 and dy != 0 or dx != 0 and dz != 0 or dy != 0 and dz != 0:
            # Explore straight paths first
            if self._jump(next_node, (dx, 0, 0), depth + 1): return next_node
            if self._jump(next_node, (0, dy, 0), depth + 1): return next_node
            if self._jump(next_node, (0, 0, dz), depth + 1): return next_node

        # Continue jumping in the current direction
        return self._jump(next_node, direction, depth + 1)

    def _is_valid_position(self, position):
        x, y, z = position
        return (0 <= x < self.grid_shape[0] and
                0 <= y < self.grid_shape[1] and
                0 <= z < self.grid_shape[2])

    def _has_forced_neighbor(self, node, direction):
        dx, dy, dz = direction
        x, y, z = node
        
        # Check for 2D forced neighbors (XY plane)
        if dx != 0 and dy != 0:
            if self.is_obstructed((x - dx, y, z)) and not self.is_obstructed((x - dx, y + dy, z)): return True
            if self.is_obstructed((x, y - dy, z)) and not self.is_obstructed((x + dx, y - dy, z)): return True
        # Check for 2D forced neighbors (XZ plane)
        if dx != 0 and dz != 0:
            if self.is_obstructed((x - dx, y, z)) and not self.is_obstructed((x - dx, y, z + dz)): return True
            if self.is_obstructed((x, y, z - dz)) and not self.is_obstructed((x + dx, y, z - dz)): return True
        # Check for 2D forced neighbors (YZ plane)
        if dy != 0 and dz != 0:
            if self.is_obstructed((x, y - dy, z)) and not self.is_obstructed((x, y - dy, z + dz)): return True
            if self.is_obstructed((x, y, z - dz)) and not self.is_obstructed((x, y + dy, z - dz)): return True
            
        return False

    def _prune_directions(self, direction):
        # Basic pruning, can be enhanced for 3D
        return self.directions

    def _reconstruct_path(self, current):
        path = [current]
        start_node = self.start
        
        while current != start_node:
            prev = self.came_from[current]
            # Interpolate between jump points
            dist = np.linalg.norm(np.array(current) - np.array(prev))
            num_points = int(dist) + 1
            interpolated_segment = [tuple(map(int, p)) for p in np.linspace(prev, current, num_points)]
            
            # Add segment in reverse order, excluding the start point (prev)
            # as it will be the end point of the next segment
            path.extend(interpolated_segment[-2::-1])
            current = prev
            
        path.reverse()
        return path