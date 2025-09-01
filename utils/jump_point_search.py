import heapq
from itertools import product
import numpy as np
import config

class JumpPointSearch:
    def __init__(self, start, goal, is_obstructed_func, heuristic, grid_shape=None):
        self.start = start
        self.goal = goal
        self.is_obstructed = is_obstructed_func
        self.heuristic = heuristic
        self.grid_shape = grid_shape or (1000, 1000, 100)  # Default bounds

        self.open_set = []
        self.open_set_map = {}
        self.came_from = {}
        self.g_score = {start: 0}
        
        self.directions = list(product([-1, 0, 1], repeat=3))
        self.directions.remove((0, 0, 0))
        
        # Safety limits
        self.MAX_JUMP_DEPTH = config.MAX_PATH_LENGTH
        self.max_iterations = 10000

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
            neighbor = (node[0] + d[0], node[1] + d[1], node[2] + d[2])
            
            if self._is_valid_position(neighbor) and not self.is_obstructed(neighbor):
                successors.add(neighbor)

            jump_point = self._jump(node, d, set(), 0)
            if jump_point:
                successors.add(jump_point)
        
        return list(successors)

    def _jump(self, node, direction, visited, depth):
        if depth > self.MAX_JUMP_DEPTH or node in visited:
            return None

        visited.add(node)
        
        next_node = (node[0] + direction[0], node[1] + direction[1], node[2] + direction[2])
        
        if not self._is_valid_position(next_node) or self.is_obstructed(next_node):
            return None
        if next_node == self.goal:
            return next_node
        if self._has_forced_neighbor(next_node, direction):
            return next_node

        # Recursive jump for diagonal movement
        dx, dy, dz = direction
        if dx != 0 and dy != 0 and dz != 0:
            if (self._jump(next_node, (dx, 0, 0), visited.copy(), depth + 1) or
                self._jump(next_node, (0, dy, 0), visited.copy(), depth + 1) or
                self._jump(next_node, (0, 0, dz), visited.copy(), depth + 1)):
                return next_node

        return self._jump(next_node, direction, visited, depth + 1)

    def _is_valid_position(self, position):
        x, y, z = position
        return (0 <= x < self.grid_shape[0] and
                0 <= y < self.grid_shape[1] and
                0 <= z < self.grid_shape[2])

    def _has_forced_neighbor(self, node, direction):
        x, y, z = node
        dx, dy, dz = direction
        
        # Simplified forced neighbor detection
        if dx != 0:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    if j != dy or k != dz:
                        check_pos = (x, y + j, z + k)
                        if (self._is_valid_position(check_pos) and 
                            self.is_obstructed(check_pos)):
                            next_forced = (x + dx, y + j, z + k)
                            if (self._is_valid_position(next_forced) and 
                                not self.is_obstructed(next_forced)):
                                return True
        return False

    def _prune_directions(self, direction):
        dx, dy, dz = direction
        pruned = [direction]
        
        if dx != 0 and dy == 0 and dz == 0:
            pruned.extend([(dx, -1, 0), (dx, 1, 0), (dx, 0, -1), (dx, 0, 1)])
        elif dy != 0 and dx == 0 and dz == 0:
            pruned.extend([(-1, dy, 0), (1, dy, 0), (0, dy, -1), (0, dy, 1)])
        elif dz != 0 and dx == 0 and dy == 0:
            pruned.extend([(-1, 0, dz), (1, 0, dz), (0, -1, dz), (0, 1, dz)])
        else:
            if dx != 0 and dy != 0:
                pruned.extend([(dx, 0, 0), (0, dy, 0), (dx, dy, 0)])
            if dx != 0 and dz != 0:
                pruned.extend([(dx, 0, 0), (0, 0, dz), (dx, 0, dz)])
            if dy != 0 and dz != 0:
                pruned.extend([(0, dy, 0), (0, 0, dz), (0, dy, dz)])
        
        return list(set(pruned))

    def _reconstruct_path(self, current):
        path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            path.append(current)
        path.reverse()
        return path