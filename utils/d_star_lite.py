# ==============================================================================
# File: utils/d_star_lite.py
# ==============================================================================
import heapq
import numpy as np
from collections import defaultdict
from itertools import product
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.coordinate_manager import CoordinateManager

class DStarLite:
    def __init__(self, start: tuple, goal: tuple, cost_map: dict, heuristic, coord_manager: 'CoordinateManager'):
        self.start = start
        self.goal = goal
        self.cost_map = cost_map
        self.heuristic = heuristic
        self.coord_manager = coord_manager

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
            if self.coord_manager.is_valid_grid_position(successor):
                successors.append(successor)
        return successors

    def _get_predecessors(self, node):
        predecessors = []
        x, y, z = node
        for dx, dy, dz in self.MOVES:
            predecessor = (x - dx, y - dy, z - dz)
            if self.coord_manager.is_valid_grid_position(predecessor):
                predecessors.append(predecessor)
        return predecessors

    def _update_node(self, node):
        if node != self.goal:
            min_rhs = float('inf')
            for s_node in self._get_successors(node):
                cost = self.heuristic.cost_between(node, s_node)
                min_rhs = min(min_rhs, cost + self.g_score[s_node])
            self.rhs_score[node] = min_rhs

        if node in self.open_set_map:
            key_to_remove = self.open_set_map.pop(node)
            self.open_set = [item for item in self.open_set if item != (key_to_remove, node)]
            heapq.heapify(self.open_set)

        if self.g_score[node] != self.rhs_score[node]:
            key = self._calculate_key(node)
            heapq.heappush(self.open_set, (key, node))
            self.open_set_map[node] = key

    def compute_shortest_path(self):
        max_iterations = 20000 
        iterations = 0
        
        while self.open_set and iterations < max_iterations:
            if not self.open_set: break
            start_key = self._calculate_key(self.start)
            top_key, _ = self.open_set[0]

            if top_key >= start_key and self.rhs_score[self.start] == self.g_score[self.start]:
                break

            iterations += 1
            key, current = heapq.heappop(self.open_set)

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
        
        for cell in cost_updates:
            self.cost_map[cell] = float('inf')
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
            if not successors: return None

            for successor in successors:
                cost = self.heuristic.cost_between(current, successor) + self.g_score[successor]
                if cost < min_cost:
                    min_cost = cost
                    next_node = successor
            
            if next_node is None:
                logging.error("D* Lite path reconstruction failed: no valid successor found.")
                return None
            
            current = next_node
            path.append(current)
            
            if len(path) > (self.coord_manager.grid_width * self.coord_manager.grid_height):
                logging.error("D* Lite path reconstruction exceeded safety limit.")
                return None
        
        return path