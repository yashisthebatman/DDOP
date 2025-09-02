import heapq
from collections import defaultdict
from itertools import product
import logging
from typing import TYPE_CHECKING, Dict, Optional, List, Tuple
# FIX: Add the missing numpy import. This resolves the NameError.
import numpy as np

if TYPE_CHECKING:
    from utils.coordinate_manager import CoordinateManager
    from utils.heuristics import HeuristicProvider

class DStarLite:
    def __init__(self, start: tuple, goal: tuple, cost_map: Dict, heuristic_provider: 'HeuristicProvider', coord_manager: 'CoordinateManager', mode: str):
        self.start = start
        self.goal = goal
        self.cost_map = cost_map
        self.heuristic = heuristic_provider.get_grid_heuristic(goal)
        self.coord_manager = coord_manager
        self.mode = mode

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
        h = self.heuristic(node)
        min_score = min(self.g_score[node], self.rhs_score[node])
        return (min_score + h + self.km, min_score)

    def _get_successors(self, node):
        successors = []
        x, y, z = node
        for dx, dy, dz in self.MOVES:
            successor = (x + dx, y + dy, z + dz)
            if self.coord_manager.is_valid_local_grid_pos(successor):
                successors.append(successor)
        return successors

    def _get_predecessors(self, node):
        predecessors = []
        x, y, z = node
        for dx, dy, dz in self.MOVES:
            predecessor = (x - dx, y - dy, z - dz)
            if self.coord_manager.is_valid_local_grid_pos(predecessor):
                predecessors.append(predecessor)
        return predecessors

    def _update_node(self, node):
        if node != self.goal:
            min_rhs = float('inf')
            for s_node in self._get_successors(node):
                cost = self._cost_between(node, s_node)
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
            
    def update_and_replan(self, new_start, cost_updates):
        self.last_start = self.start
        self.start = new_start
        self.km += self.heuristic(self.last_start)
        
        for cell in cost_updates:
            self.cost_map[cell] = float('inf')
            self._update_node(cell)
            for pred in self._get_predecessors(cell):
                 self._update_node(pred)
        return self.compute_shortest_path()

    def get_path(self) -> Optional[List[Tuple]]:
        if self.g_score[self.start] == float('inf'):
            return None
        
        path = [self.start]
        current = self.start
        max_path_len = self.coord_manager.grid_shape[0] * self.coord_manager.grid_shape[1] * self.coord_manager.grid_shape[2]
        
        while current != self.goal:
            min_cost = float('inf')
            next_node = None
            
            for successor in self._get_successors(current):
                cost = self._cost_between(current, successor) + self.g_score[successor]
                if cost < min_cost:
                    min_cost = cost
                    next_node = successor
            
            if next_node is None:
                logging.error("D* Lite path reconstruction failed: no valid successor found.")
                return None
            
            current = next_node
            path.append(current)
            
            if len(path) > max_path_len:
                logging.error("D* Lite path reconstruction exceeded safety limit.")
                return None
        
        return path

    def _cost_between(self, n1: Tuple, n2: Tuple) -> float:
        if n2 in self.cost_map:
            return self.cost_map[n2]
        return np.linalg.norm(np.array(n1) - np.array(n2))