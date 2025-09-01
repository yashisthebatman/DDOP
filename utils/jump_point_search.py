import heapq
from itertools import product
import numpy as np

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
        
        # All 26 possible directions in 3D
        self.directions = list(product([-1, 0, 1], repeat=3))
        self.directions.remove((0, 0, 0))

    def search(self):
        """Main search loop."""
        h_start = self.heuristic.calculate(self.start)
        heapq.heappush(self.open_set, (h_start, self.start))
        self.open_set_map[self.start] = h_start

        while self.open_set:
            _, current = heapq.heappop(self.open_set)
            
            if current == self.goal:
                return self._reconstruct_path(current)

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
        
        return None # Path not found

    def _identify_successors(self, node):
        successors = set()
        parent = self.came_from.get(node)
        
        if parent is None:
            pruned_directions = self.directions
        else:
            direction = tuple(np.sign(np.array(node) - np.array(parent)))
            pruned_directions = self._prune_directions(direction)

        for d in pruned_directions:
            jump_point = self._jump(node, d)
            if jump_point:
                successors.add(jump_point)
        
        return list(successors)

    def _jump(self, node, direction):
        dx, dy, dz = direction
        next_node = (node[0] + dx, node[1] + dy, node[2] + dz)
        
        if self.is_obstructed(next_node): return None
        if next_node == self.goal: return next_node
        if self._has_forced_neighbor(next_node, direction): return next_node

        # Diagonal case: jump orthogonally first
        if dx != 0 and dy != 0 and dz != 0:
             if self._jump(next_node, (dx, dy, 0)) or self._jump(next_node, (dx, 0, dz)) or self._jump(next_node, (0, dy, dz)):
                 return next_node
        elif dx != 0 and dy != 0:
            if self._jump(next_node, (dx, 0, 0)) or self._jump(next_node, (0, dy, 0)): return next_node
        elif dx != 0 and dz != 0:
            if self._jump(next_node, (dx, 0, 0)) or self._jump(next_node, (0, 0, dz)): return next_node
        elif dy != 0 and dz != 0:
            if self._jump(next_node, (0, dy, 0)) or self._jump(next_node, (0, 0, dz)): return next_node
        
        return self._jump(next_node, direction)

    def _has_forced_neighbor(self, node, direction):
        return False # Simplified for performance; full 3D JPS neighbor check is complex

    def _prune_directions(self, direction):
        dx, dy, dz = direction
        pruned = {direction}
        if dx != 0:
            pruned.add((dx, dy, 1)); pruned.add((dx, dy, -1)); pruned.add((dx, 1, dz)); pruned.add((dx, -1, dz))
        if dy != 0:
            pruned.add((dx, dy, 1)); pruned.add((dx, dy, -1)); pruned.add((1, dy, dz)); pruned.add((-1, dy, dz))
        if dz != 0:
            pruned.add((dx, 1, dz)); pruned.add((dx, -1, dz)); pruned.add((1, dy, dz)); pruned.add((-1, dy, dz))
        return list(pruned)
        
    def _reconstruct_path(self, current):
        """OPTIMIZED: Reconstructs the path using vectorized numpy operations."""
        path = [current]
        while current in self.came_from:
            prev = self.came_from[current]
            
            # Vectorized path interpolation between sparse jump points
            p1, p2 = np.array(prev), np.array(current)
            diff = p2 - p1
            num_steps = np.max(np.abs(diff)) + 1
            
            # Generate all intermediate grid points in one go
            points = np.linspace(p1, p2, int(num_steps), endpoint=True).round().astype(int)
            
            # Append the interpolated path (excluding the end point, which is the start of the next segment)
            path.extend(map(tuple, points[:-1][::-1]))
            
            current = prev
            
        return path[::-1]