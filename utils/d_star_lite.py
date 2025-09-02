import heapq
from collections import defaultdict
from itertools import product
import logging
from typing import TYPE_CHECKING, Dict, Optional, List, Tuple
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
        self.MOVES = [move for move in product([-1, 0, 1], repeat=3) if move != (0, 0, 0)]
        
        self.rhs_score[self.goal] = 0
        key = self._calculate_key(self.goal)
        heapq.heappush(self.open_set, (key, self.goal))
        self.open_set_map[self.goal] = key
        
    def _calculate_key(self, node):
        h = self.heuristic(node)
        min_score = min(self.g_score[node], self.rhs_score[node])
        return (min_score + h + self.km, min_score)

    def _get_successors(self, node: Tuple) -> List[Tuple]:
        """Gets all valid neighboring nodes."""
        for dx, dy, dz in self.MOVES:
            successor = (node[0] + dx, node[1] + dy, node[2] + dz)
            if self.coord_manager.is_valid_local_grid_pos(successor):
                yield successor

    def _get_predecessors(self, node: Tuple) -> List[Tuple]:
        """Gets all valid nodes that could lead to the current node."""
        for dx, dy, dz in self.MOVES:
            predecessor = (node[0] - dx, node[1] - dy, node[2] - dz)
            if self.coord_manager.is_valid_local_grid_pos(predecessor):
                yield predecessor

    def _update_node(self, node):
        if node != self.goal:
            self.rhs_score[node] = min(
                (self._cost_between(node, s) + self.g_score[s] for s in self._get_successors(node)),
                default=float('inf')
            )
        
        if node in self.open_set_map:
            # Node is in the queue, we can remove it logically by just popping from map
            self.open_set_map.pop(node)

        if self.g_score[node] != self.rhs_score[node]:
            key = self._calculate_key(node)
            heapq.heappush(self.open_set, (key, node))
            self.open_set_map[node] = key

    def compute_shortest_path(self):
        max_iterations = 30000
        while self.open_set and max_iterations > 0:
            max_iterations -= 1
            top_key = self.open_set[0][0]
            start_key = self._calculate_key(self.start)
            
            if top_key >= start_key and self.rhs_score[self.start] == self.g_score[self.start]:
                return
            
            key, current = heapq.heappop(self.open_set)
            
            if current not in self.open_set_map or self.open_set_map[current] != key:
                continue # Stale entry
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

    def update_and_replan(self, new_start, cost_updates):
        self.last_start = self.start
        self.start = new_start
        self.km += self.heuristic(self.last_start)
        
        for cell in cost_updates:
            self.cost_map[cell] = float('inf')
            self._update_node(cell)
            for pred in self._get_predecessors(cell):
                self._update_node(pred)
        self.compute_shortest_path()

    def get_path(self) -> Optional[List[Tuple]]:
        if self.g_score[self.start] == float('inf'):
            return None
        path = [self.start]
        current = self.start
        max_path_len = self.coord_manager.grid_width * self.coord_manager.grid_height * 2
        
        while current != self.goal:
            if len(path) > max_path_len: return None # Safety break
            
            next_node = min(
                self._get_successors(current),
                key=lambda s: self._cost_between(current, s) + self.g_score[s],
                default=None
            )
            if next_node is None: return None
            
            current = next_node
            path.append(current)
        return path

    def _cost_between(self, n1: Tuple, n2: Tuple) -> float:
        return self.cost_map.get(n2, np.linalg.norm(np.array(n1) - np.array(n2)))