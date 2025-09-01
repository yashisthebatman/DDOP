import heapq
import numpy as np
from collections import defaultdict
from itertools import product

class DStarLite:
    def __init__(self, start, goal, cost_map, heuristic):
        self.start = start
        self.goal = goal
        self.cost_map = cost_map 
        self.heuristic = heuristic

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
            self.open_set = [(k, n) for k, n in self.open_set if n != node]
            heapq.heapify(self.open_set)
            del self.open_set_map[node]

        if self.g_score[node] != self.rhs_score[node]:
            key = self._calculate_key(node)
            heapq.heappush(self.open_set, (key, node))
            self.open_set_map[node] = key

    def compute_shortest_path(self):
        while self.open_set:
            start_key = self._calculate_key(self.start)
            if not self.open_set or self.open_set[0][0] >= start_key and self.rhs_score[self.start] == self.g_score[self.start]:
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

    def update_and_replan(self, new_start, cost_updates):
        self.last_start = self.start
        self.start = new_start
        self.km += self.heuristic.calculate(self.last_start)
        
        for cell, new_cost in cost_updates:
            # Update the underlying cost provider (the planner's cost map)
            self.heuristic.planner.cost_map[cell] = new_cost
            
            # --- BUG FIX: Update the now-obstructed cell itself, and its predecessors ---
            # This ensures the algorithm knows the cell is now unreachable.
            self._update_node(cell) 
            for p_node in self._get_predecessors(cell):
                self._update_node(p_node)
        
        return self.compute_shortest_path()

    def get_path(self):
        if self.g_score[self.start] == float('inf'): return None
        path = [self.start]
        current = self.start
        while current != self.goal:
            successors = list(self._get_successors(current))
            if not successors: return None
            best_s = min(successors, key=lambda s: self.heuristic.cost_between(current, s) + self.g_score[s])
            if self.g_score[best_s] == float('inf'): return None
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