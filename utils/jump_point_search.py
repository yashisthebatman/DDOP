

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
        self.max_iterations = 20000

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

        # Diagonal jump recursion
        dx, dy, dz = direction
        if dx != 0 and dy != 0 or dx != 0 and dz != 0 or dy != 0 and dz != 0:
            # Check for jump points in cardinal components of the diagonal move
            if self._jump(next_node, (dx, 0, 0), visited.copy(), depth + 1): return next_node
            if self._jump(next_node, (0, dy, 0), visited.copy(), depth + 1): return next_node
            if self._jump(next_node, (0, 0, dz), visited.copy(), depth + 1): return next_node

        return self._jump(next_node, direction, visited, depth + 1)

    def _is_valid_position(self, position):
        x, y, z = position
        return (0 <= x < self.grid_shape[0] and
                0 <= y < self.grid_shape[1] and
                0 <= z < self.grid_shape[2])

    def _has_forced_neighbor(self, node, direction):
        """Systematic check for forced neighbors in 3D."""
        dx, dy, dz = direction
        x, y, z = node

        # Normalize direction for checks
        ndx = dx if dx == 0 else dx // abs(dx)
        ndy = dy if dy == 0 else dy // abs(dy)
        ndz = dz if dz == 0 else dz // abs(dz)

        # Cardinal moves have no forced neighbors
        if (abs(ndx) + abs(ndy) + abs(ndz)) == 1:
            return False

        # Check for obstacles that would force a turn
        # For a move (dx, dy, dz), we look for obstacles at (x-dx, y, z), (x, y-dy, z), etc.
        # that have an open space next to them in the direction of travel.

        if ndx != 0 and ndy != 0: # Planar diagonal move on XY plane
            if self.is_obstructed((x - ndx, y, z)) and not self.is_obstructed((x - ndx, y + ndy, z)): return True
            if self.is_obstructed((x, y - ndy, z)) and not self.is_obstructed((x + ndx, y - ndy, z)): return True
        
        if ndx != 0 and ndz != 0: # Planar diagonal move on XZ plane
            if self.is_obstructed((x - ndx, y, z)) and not self.is_obstructed((x - ndx, y, z + ndz)): return True
            if self.is_obstructed((x, y, z - ndz)) and not self.is_obstructed((x + ndx, y, z - ndz)): return True

        if ndy != 0 and ndz != 0: # Planar diagonal move on YZ plane
            if self.is_obstructed((x, y - ndy, z)) and not self.is_obstructed((x, y - ndy, z + ndz)): return True
            if self.is_obstructed((x, y, z - ndz)) and not self.is_obstructed((x, y + ndy, z - ndz)): return True

        return False

    def _prune_directions(self, direction):
        dx, dy, dz = direction
        pruned = {direction}
        
        # Natural neighbors
        if dx != 0: pruned.add((dx, 0, 0))
        if dy != 0: pruned.add((0, dy, 0))
        if dz != 0: pruned.add((0, 0, dz))
        if dx != 0 and dy != 0: pruned.add((dx, dy, 0))
        if dx != 0 and dz != 0: pruned.add((dx, 0, dz))
        if dy != 0 and dz != 0: pruned.add((0, dy, dz))

        # Forced neighbors
        if dx != 0 and dy != 0: # Moving in XY plane
            if self.is_obstructed((direction[0], -direction[1], 0)): pruned.add((dx, -dy, 0))
            if self.is_obstructed((-direction[0], direction[1], 0)): pruned.add((-dx, dy, 0))
        
        # Add similar checks for XZ and YZ planes if needed for more aggressive pruning

        return list(pruned)

    def _reconstruct_path(self, current):
        path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            path.append(current)
        path.reverse()
        return path