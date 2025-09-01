# utils/d_star_lite.py
import heapq
import numpy as np
from collections import defaultdict
from itertools import product  # <-- ADD THIS LINE

class DStarLite:
    def __init__(self, start, goal, cost_map, heuristic):
        self.start = start
        self.goal = goal
        self.cost_map = cost_map
        self.heuristic = heuristic

        self.g_score = defaultdict(lambda: float('inf'))
        self.rhs_score = defaultdict(lambda: float('inf'))
        self.open_set = []  # Priority Queue (U)
        self.open_set_map = {}
        
        self.km = 0  # Key modifier
        self.last_start = start

        self.rhs_score[self.goal] = 0
        key = self._calculate_key(self.goal)
        heapq.heappush(self.open_set, (key, self.goal))
        self.open_set_map[self.goal] = key
        
    def _calculate_key(self, node):
        h = self.heuristic.calculate(node)
        g = self.g_score[node]
        rhs = self.rhs_score[node]
        min_score = min(g, rhs)
        return (min_score + h + self.km, min_score)

    def _update_node(self, node):
        if node != self.goal:
            min_rhs = float('inf')
            for s_node in self._get_successors(node):
                cost = self.heuristic.cost_between(node, s_node)
                min_rhs = min(min_rhs, cost + self.g_score[s_node])
            self.rhs_score[node] = min_rhs

        if node in self.open_set_map:
            del self.open_set_map[node]

        if self.g_score[node] != self.rhs_score[node]:
            key = self._calculate_key(node)
            heapq.heappush(self.open_set, (key, node))
            self.open_set_map[node] = key

    def compute_shortest_path(self):
        while self.open_set:
            start_key = self._calculate_key(self.start)
            top_key, _ = self.open_set[0]
            if top_key >= start_key and self.rhs_score[self.start] == self.g_score[self.start]:
                break

            key, current = heapq.heappop(self.open_set)
            if current not in self.open_set_map or self.open_set_map[current] < key:
                continue
            del self.open_set_map[current]

            old_key = self._calculate_key(current) # This should be key, not a new calculation
            if old_key < key:
                heapq.heappush(self.open_set, (key, current)) # Push with the key it was popped with
                self.open_set_map[current] = key
            elif self.g_score[current] > self.rhs_score[current]:
                self.g_score[current] = self.rhs_score[current]
                for p_node in self._get_predecessors(current):
                    self._update_node(p_node)
            else:
                self.g_score[current] = float('inf')
                self._update_node(current)
                for p_node in self._get_predecessors(current):
                    self._update_node(p_node)
        return self.get_path()

    def update_and_replan(self, new_start, cost_updates):
        self.last_start = self.start
        self.start = new_start
        
        self.km += self.heuristic.calculate(self.last_start)
        
        for cell, new_cost in cost_updates:
            # Update costs for affected edges by updating the predecessors of the changed cell
            for p_node in self._get_predecessors(cell):
                self._update_node(p_node)
        
        return self.compute_shortest_path()

    def get_path(self):
        if self.g_score[self.start] == float('inf'):
            return None
        path = [self.start]
        current = self.start
        while current != self.goal:
            successors = list(self._get_successors(current))
            if not successors: return None # Stuck
            best_s = min(successors, key=lambda s: self.heuristic.cost_between(current, s) + self.g_score[s])
            current = best_s
            path.append(current)
        return path

    def _get_neighbors(self, node):
        moves = list(product([-1, 0, 1], repeat=3))
        moves.remove((0, 0, 0))
        for move in moves:
            yield (node[0] + move[0], node[1] + move[1], node[2] + move[2])

    def _get_predecessors(self, node):
        return self._get_neighbors(node)

    def _get_successors(self, node):
        return self._get_neighbors(node)