import heapq
import logging
from collections import defaultdict
from itertools import product
from typing import Dict, List, Tuple, Optional
import numpy as np
from utils.coordinate_manager import CoordinateManager
from utils.heuristics import HeuristicProvider

class DStarLite:
    """A robust implementation of the D* Lite algorithm for single-agent replanning."""
    def __init__(self, start: tuple, goal: tuple, cost_map: Dict, heuristic_provider: HeuristicProvider, coord_manager: CoordinateManager):
        self.start, self.goal = start, goal
        self.cost_map = cost_map
        self.heuristic = heuristic_provider.get_grid_heuristic(goal)
        self.coord_manager = coord_manager

        self.g_score = defaultdict(lambda: float('inf'))
        self.rhs_score = defaultdict(lambda: float('inf'))
        
        self.open_set = [] 
        self.open_set_map = {} 

        self.km = 0.0
        self.MOVES = [move for move in product([-1, 0, 1], repeat=3) if move != (0, 0, 0)]
        
        self.rhs_score[self.goal] = 0
        self._update_queue(self.goal)

    def _calculate_key(self, node: Tuple) -> Tuple[float, float]:
        h = self.heuristic(node)
        return (min(self.g_score[node], self.rhs_score[node]) + h + self.km,
                min(self.g_score[node], self.rhs_score[node]))

    def _update_queue(self, node: tuple):
        if node in self.open_set_map:
            del self.open_set_map[node]
        if self.g_score[node] != self.rhs_score[node]:
            key = self._calculate_key(node)
            heapq.heappush(self.open_set, (key, node))
            self.open_set_map[node] = key

    def _update_node(self, node: tuple):
        if node != self.goal:
            self.rhs_score[node] = min((self._cost_between(node, s) + self.g_score[s]
                                      for s in self._get_successors(node)), default=float('inf'))
        self._update_queue(node)

    def compute_shortest_path(self):
        while self.open_set:
            if not self.open_set_map: break
            top_key = self.open_set[0][0]
            start_key = self._calculate_key(self.start)
            if top_key >= start_key and self.rhs_score[self.start] == self.g_score[self.start]:
                break
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

    def update_and_replan(self, new_start: tuple, cost_updates: Dict):
        self.km += self.heuristic(self.start)
        self.start = new_start
        for changed_node, new_cost in cost_updates.items():
            self.cost_map[changed_node] = new_cost
            self._update_node(changed_node)
            for p_node in self._get_predecessors(changed_node):
                 self._update_node(p_node)
        self.compute_shortest_path()

    def get_path(self) -> Optional[List[Tuple]]:
        if self.g_score[self.start] == float('inf'):
            logging.warning("D* Lite: No path found.")
            return None
        path = [self.start]
        current = self.start
        while current != self.goal:
            if len(path) > 2000:
                logging.error("D* Lite path reconstruction exceeded max length.")
                return None
            successors = list(self._get_successors(current))
            if not successors:
                logging.error(f"D* Lite path reconstruction failed: no successors for {current}")
                return None
            current = min(successors, key=lambda s: self._cost_between(current, s) + self.g_score[s])
            path.append(current)
        return path

    def _cost_between(self, n1: Tuple, n2: Tuple) -> float:
        if self.cost_map.get(n1) == float('inf') or self.cost_map.get(n2) == float('inf'):
            return float('inf')
        return np.linalg.norm(np.array(n1) - np.array(n2))

    def _get_successors(self, node: Tuple):
        for move in self.MOVES:
            succ = tuple(a + b for a, b in zip(node, move))
            if self.coord_manager.is_valid_local_grid_pos(succ): yield succ

    def _get_predecessors(self, node: Tuple):
        for move in self.MOVES:
            pred = tuple(a - b for a, b in zip(node, move))
            if self.coord_manager.is_valid_local_grid_pos(pred): yield pred