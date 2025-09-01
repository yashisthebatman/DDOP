# utils/d_star_lite.py (Corrected and Optimized)

import heapq
from collections import defaultdict
from itertools import product
import numpy as np

class DStarLite:
    def __init__(self, start, goal, cost_map, heuristic):
        self.start = start
        self.goal = goal
        self.cost_map = cost_map
        self.heuristic = heuristic

        self.g_score = defaultdict(lambda: float('inf'))
        self.rhs_score = defaultdict(lambda: float('inf'))
        self.open_set = [] # Priority queue
        self.open_set_map = {} # For quick lookups and efficient removal
        
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

    def _update_node(self, node):
        if node != self.goal:
            min_rhs = float('inf')
            for s_node in self._get_successors(node):
                cost = self.heuristic.cost_between(node, s_node)
                min_rhs = min(min_rhs, cost + self.g_score[s_node])
            self.rhs_score[node] = min_rhs

        # --- PERFORMANCE FIX: Efficient "lazy removal" from priority queue ---
        if node in self.open_set_map:
            # Mark as invalid instead of rebuilding the heap
            self.open_set_map.pop(node)

        if self.g_score[node] != self.rhs_score[node]:
            key = self._calculate_key(node)
            heapq.heappush(self.open_set, (key, node))
            self.open_set_map[node] = key

    def compute_shortest_path(self):
        while self.open_set:
            # Pop valid nodes, discarding "lazily removed" ones
            while self.open_set and self.open_set[0][1] not in self.open_set_map:
                heapq.heappop(self.open_set)
            if not self.open_set: break

            start_key = self._calculate_key(self.start)
            if self.open_set[0][0] >= start_key and self.rhs_score[self.start] == self.g_score[self.start]:
                break

            key, current = heapq.heappop(self.open_set)
            if current in self.open_set_map:
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
        return self.get_path()
        
    def get_path(self):
        if self.g_score[self.start] == float('inf'): return None
        path = [self.start]
        current = self.start
        while current != self.goal:
            successors = list(self._get_successors(current))
            if not successors: return None
            
            best_s = min(successors, key=lambda s: self.heuristic.cost_between(current, s) + self.g_score[s])
            
            # --- CRITICAL BUG FIX: Check if the best path is unreachable ---
            if self.g_score[best_s] == float('inf'):
                return None # No valid path found, fail gracefully
                
            current = best_s
            path.append(current)
        return path

    def _get_neighbors(self, node):
        for move in self.MOVES:
            yield (node[0] + move[0], node[1] + move[1], node[2] + move[2])

    def _get_predecessors(self, node):
        return self._get_neighbors(node)

    def _get_successors(self, node):
        return self._get_neighbors(node)