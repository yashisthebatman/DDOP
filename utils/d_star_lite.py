
import heapq
import numpy as np
from collections import defaultdict
from itertools import product

class DStarLite:
    def __init__(self, start, goal, cost_map, heuristic, grid_dims):
        self.start = start
        self.goal = goal
        self.cost_map = cost_map
        self.heuristic = heuristic
        self.grid_width, self.grid_height, self.grid_depth = grid_dims

        self.g_score = defaultdict(lambda: float('inf'))
        self.rhs_score = defaultdict(lambda: float('inf'))
        self.open_set = []
        self.open_set_map = {}
        
        self.km = 0
        self.last_start = start
        
        moves = list(product([-1, 0, 1], repeat=3))
        moves.remove((0, 0, 0))
        self.MOVES = moves

        self.rhs_score[self.goal] = 0
        key = self._calculate_key(self.goal)
        heapq.heappush(self.open_set, (key, self.goal))
        self.open_set_map[self.goal] = key
        
    def _calculate_key(self, node):
        h = self.heuristic.calculate(node)
        min_score = min(self.g_score[node], self.rhs_score[node])
        return (min_score + h + self.km, min_score)

    def _get_successors(self, node):
        successors = []
        x, y, z = node
        for dx, dy, dz in self.MOVES:
            successor = (x + dx, y + dy, z + dz)
            if self._is_valid_node(successor):
                successors.append(successor)
        return successors

    def _get_predecessors(self, node):
        predecessors = []
        x, y, z = node
        for dx, dy, dz in self.MOVES:
            predecessor = (x - dx, y - dy, z - dz)
            if self._is_valid_node(predecessor):
                predecessors.append(predecessor)
        return predecessors

    def _is_valid_node(self, node):
        x, y, z = node
        return (0 <= x < self.grid_width and 
                0 <= y < self.grid_height and 
                0 <= z < self.grid_depth)

    def _update_node(self, node):
        if node != self.goal:
            min_rhs = float('inf')
            for s_node in self._get_successors(node):
                cost = self.heuristic.cost_between(node, s_node)
                min_rhs = min(min_rhs, cost + self.g_score[s_node])
            self.rhs_score[node] = min_rhs

        if node in self.open_set_map:
            # Remove from heap is tricky, so we mark as invalid and ignore later
            self.open_set_map.pop(node)

        if self.g_score[node] != self.rhs_score[node]:
            key = self._calculate_key(node)
            heapq.heappush(self.open_set, (key, node))
            self.open_set_map[node] = key

    def compute_shortest_path(self):
        max_iterations = 20000 # Increased for safety
        iterations = 0
        
        while self.open_set and iterations < max_iterations:
            start_key = self._calculate_key(self.start)
            
            # Peek at top of heap
            top_key, _ = self.open_set[0]

            if top_key >= start_key and self.rhs_score[self.start] == self.g_score[self.start]:
                break

            iterations += 1
            
            key, current = heapq.heappop(self.open_set)

            # If node is stale (already processed with a better key), skip it
            if current not in self.open_set_map or self.open_set_map[current] != key:
                continue
            
            del self.open_set_map[current]

            if self.g_score[current] > self.rhs_score[current]:
                self.g_score[current] = self.rhs_score[current]
                for p_node in self._get_predecessors(current):
                    self._update_node(p_node)
            else:
                self.g_score[current] = float('inf')
                self._update_node(current)
                for p_node in self._get_predecessors(current):
                    self._update_node(p_node)
        
        if iterations >= max_iterations:
            logging.warning("D* Lite reached max iterations.")
            
        return self.get_path()

    def update_and_replan(self, new_start, cost_updates):
        self.last_start = self.start
        self.start = new_start
        self.km += self.heuristic.calculate(self.last_start)
        
        for cell, new_cost in cost_updates:
            # This is a simplified cost update logic.
            # A full implementation would need to check c_old vs c_new.
            self.heuristic.planner.cost_map[cell] = new_cost
            self._update_node(cell)
            for pred in self._get_predecessors(cell):
                self._update_node(pred)
        
        return self.compute_shortest_path()

    def get_path(self):
        if self.g_score[self.start] == float('inf'):
            return None
        
        path = [self.start]
        current = self.start
        
        while current != self.goal:
            min_cost = float('inf')
            next_node = None
            
            successors = self._get_successors(current)
            if not successors:
                return None # No path forward

            for successor in successors:
                cost = self.heuristic.cost_between(current, successor) + self.g_score[successor]
                if cost < min_cost:
                    min_cost = cost
                    next_node = successor
            
            if next_node is None:
                return None # Should not happen if g_score is not inf
            
            current = next_node
            path.append(current)
            
            if len(path) > (self.grid_width * self.grid_height): # Safety limit
                logging.error("D* Lite path reconstruction exceeded safety limit.")
                return None
        
        return path