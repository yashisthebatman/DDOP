# utils/jump_point_search.py (Corrected and Stabilized)

import heapq
from itertools import product
import numpy as np
import config

class JumpPointSearch:
    def __init__(self, start, goal, is_obstructed_func, heuristic):
        self.start = start
        self.goal = goal
        self.is_obstructed = is_obstructed_func
        self.heuristic = heuristic

        self.open_set = []
        self.open_set_map = {}
        self.came_from = {}
        self.g_score = {start: 0}
        
        self.directions = list(product([-1, 0, 1], repeat=3))
        self.directions.remove((0, 0, 0))
        
        # --- CRITICAL BUG FIX: Set a max recursion depth to prevent stack overflow ---
        self.MAX_JUMP_DEPTH = config.MAX_PATH_LENGTH 

    def search(self):
        if self.start == self.goal:
            return [self.start]
            
        h_start = self.heuristic.calculate(self.start)
        heapq.heappush(self.open_set, (h_start, self.start))
        self.open_set_map[self.start] = h_start

        while self.open_set:
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
            direction = tuple(np.sign(np.array(node) - np.array(parent)))
            pruned_directions = self._prune_directions(direction)
        
        for d in pruned_directions:
            # Add direct walkable neighbors for robustness
            neighbor = (node[0] + d[0], node[1] + d[1], node[2] + d[2])
            if not self.is_obstructed(neighbor):
                successors.add(neighbor)

            # Search for jump points
            jump_point = self._jump(node, d, set(), 0) # Pass initial visited set and depth
            if jump_point:
                successors.add(jump_point)
        
        return list(successors)

    def _jump(self, node, direction, visited, depth):
        # --- CRITICAL BUG FIX: Added depth and visited check ---
        if depth > self.MAX_JUMP_DEPTH or node in visited:
            return None

        visited.add(node)
        
        next_node = (node[0] + direction[0], node[1] + direction[1], node[2] + direction[2])
        
        if self.is_obstructed(next_node): return None
        if next_node == self.goal: return next_node
        if self._has_forced_neighbor(next_node, direction): return next_node

        # Recursive calls with updated depth and visited set
        if direction[0] != 0 and direction[1] != 0: # Diagonal 2D
            if self._jump(next_node, (direction[0], 0, 0), visited.copy(), depth + 1) or \
               self._jump(next_node, (0, direction[1], 0), visited.copy(), depth + 1):
                return next_node

        return self._jump(next_node, direction, visited, depth + 1)
    
    def _has_forced_neighbor(self, node, direction):
        # Placeholder for forced neighbor logic (essential for true JPS)
        return False

    def _prune_directions(self, direction):
        # Simplified pruning for demonstration
        return [direction]

    def _reconstruct_path(self, current):
        # Optimized path reconstruction
        path = []
        while current in self.came_from:
            prev = self.came_from[current]
            p1, p2 = np.array(prev), np.array(current)
            diff = p2 - p1
            num_steps = np.max(np.abs(diff))
            points = np.linspace(p1, p2, int(num_steps) + 1, endpoint=True).round().astype(int)
            path.extend(map(tuple, points[:-1]))
            current = prev
        path.append(self.start)
        return path[::-1]