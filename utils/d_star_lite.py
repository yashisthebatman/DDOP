# ==============================================================================
# File: utils/d_star_lite.py
# ==============================================================================
import heapq
import logging
from collections import defaultdict
from itertools import product
from typing import Dict, List, Tuple, Optional
import numpy as np
from utils.coordinate_manager import CoordinateManager
from utils.heuristics import HeuristicProvider

class DStarLite:
    def __init__(self, start: tuple, goal: tuple, cost_map: Dict, heuristic_provider: HeuristicProvider, coord_manager: CoordinateManager, mode: str):
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
        
        self.rhs_score[self.goal] = 0.0
        self._push_to_openset(self.goal)

    def _calculate_key(self, node: Tuple) -> Tuple[float, float]:
        h = self.heuristic(node)
        min_score = min(self.g_score[node], self.rhs_score[node])
        return (min_score + h + self.km, min_score)
    
    def _push_to_openset(self, node: Tuple):
        # A node is only pushed if it's inconsistent.
        # Removing from the map is handled before calling this in _update_node.
        key = self._calculate_key(node)
        heapq.heappush(self.open_set, (key, node))
        self.open_set_map[node] = key

    def _update_node(self, node: Tuple):
        if node in self.open_set_map:
            # We are about to re-calculate its priority, so we can remove it from map control.
            # The stale entry will be ignored by the lazy-removal check in the main loop.
            del self.open_set_map[node]

        if node != self.goal:
            self.rhs_score[node] = min((self._cost_between(node, s) + self.g_score[s] for s in self._get_successors(node)), default=float('inf'))
        
        if self.g_score[node] != self.rhs_score[node]:
            self._push_to_openset(node)

    def compute_shortest_path(self):
        # FIX: This is a robust, textbook implementation of the D* Lite main loop.
        logging.debug(f"Starting D* Lite. Goal={self.goal}, Start={self.start}. Initial Open Set size: {len(self.open_set)}")
        
        while self.open_set:
            start_key = self._calculate_key(self.start)
            if not self.open_set or self.open_set[0][0] >= start_key and self.rhs_score[self.start] == self.g_score[self.start]:
                logging.debug("Convergence condition met. Exiting.")
                break
            
            key, current = heapq.heappop(self.open_set)

            if current not in self.open_set_map or self.open_set_map[current] > key:
                continue # Stale entry
            
            del self.open_set_map[current]

            logging.debug(f"Processing node {current}: g={self.g_score[current]:.2f}, rhs={self.rhs_score[current]:.2f}, key={key}")

            if self.g_score[current] > self.rhs_score[current]:
                logging.debug(f"  -> Node is UNDERconsistent. Setting g = rhs.")
                self.g_score[current] = self.rhs_score[current]
                for p_node in self._get_predecessors(current):
                    self._update_node(p_node)
            else:
                logging.debug(f"  -> Node is OVERconsistent. Setting g = inf.")
                self.g_score[current] = float('inf')
                self._update_node(current)
                for p_node in self._get_predecessors(current):
                    self._update_node(p_node)

    def update_and_replan(self, new_start: Tuple, cost_updates: Dict):
        self.km += self.heuristic(self.start)
        self.start = new_start
        for cell, cost in cost_updates.items():
            self.cost_map[cell] = cost
            self._update_node(cell)
            for pred in self._get_predecessors(cell): self._update_node(pred)
        self.compute_shortest_path()

    def get_path(self) -> Optional[List[Tuple]]:
        if self.g_score[self.start] == float('inf'):
            logging.warning("D* Lite: No path found, g_score for start is infinity.")
            return None
        path, current = [self.start], self.start
        while current != self.goal:
            if len(path) > self.coord_manager.grid_width * self.coord_manager.grid_height: 
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
        return self.cost_map.get(n2, np.linalg.norm(np.array(n1) - np.array(n2)))

    def _get_successors(self, node: Tuple):
        for move in self.MOVES:
            succ = tuple(a + b for a, b in zip(node, move))
            if self.coord_manager.is_valid_local_grid_pos(succ): yield succ

    def _get_predecessors(self, node: Tuple):
        for move in self.MOVES:
            pred = tuple(a - b for a, b in zip(node, move))
            if self.coord_manager.is_valid_local_grid_pos(pred): yield pred