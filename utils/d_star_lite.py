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
        
        # The open_set is a priority queue (heap)
        self.open_set = [] 
        # The open_set_map is used for quick lookups and updates
        self.open_set_map = {} 

        self.km = 0.0
        self.MOVES = [move for move in product([-1, 0, 1], repeat=3) if move != (0, 0, 0)]
        
        self.rhs_score[self.goal] = 0.0
        key = self._calculate_key(self.goal)
        heapq.heappush(self.open_set, (key, self.goal))
        self.open_set_map[self.goal] = key

    def _calculate_key(self, node: Tuple) -> Tuple[float, float]:
        h = self.heuristic(node)
        min_score = min(self.g_score[node], self.rhs_score[node])
        return (min_score + h + self.km, min_score)

    def _remove_from_openset(self, node: Tuple):
        """Removes a node from the open set map."""
        if node in self.open_set_map:
            del self.open_set_map[node]
            # Note: We use lazy removal for the heap itself. Stale entries will be ignored.

    def _update_node(self, node: Tuple):
        """Updates the rhs-value of a node and its priority in the open set."""
        # The node is inconsistent, so remove it from the open set if it's there
        self._remove_from_openset(node)

        # Update the rhs-value if it's not the goal
        if node != self.goal:
            self.rhs_score[node] = min(
                (self._cost_between(node, s) + self.g_score[s] for s in self._get_successors(node)), 
                default=float('inf')
            )
        
        # If the node is now inconsistent, add it back to the open set with its new priority
        if self.g_score[node] != self.rhs_score[node]:
            key = self._calculate_key(node)
            heapq.heappush(self.open_set, (key, node))
            self.open_set_map[node] = key

    def compute_shortest_path(self):
        """Computes the shortest path from the start to the goal."""
        while self.open_set:
            # Get the top key from the priority queue without removing the item
            top_key = self.open_set[0][0]
            start_key = self._calculate_key(self.start)

            # Termination condition: The path is optimal when the start node's priority
            # is better than or equal to the best node in the queue, AND the start is consistent.
            if top_key >= start_key and self.rhs_score[self.start] == self.g_score[self.start]:
                break

            # Pop the node with the smallest key
            key, current = heapq.heappop(self.open_set)

            # Lazy removal: if the key we popped is stale (worse than the current best), skip it.
            if current in self.open_set_map and key > self.open_set_map[current]:
                continue
            
            # This node is being processed, so remove it from map control.
            if current in self.open_set_map:
                del self.open_set_map[current]

            # Process the node based on whether it's under-consistent or over-consistent
            if self.g_score[current] > self.rhs_score[current]:
                # Under-consistent: This node has found a better path. Update its g-score.
                self.g_score[current] = self.rhs_score[current]
                # Propagate this new, better cost to its predecessors.
                for p_node in self._get_predecessors(current):
                    self._update_node(p_node)
            else:
                # Over-consistent: The path through this node has become worse.
                # Set its g-score to infinity and update it and its predecessors.
                self.g_score[current] = float('inf')
                self._update_node(current)
                for p_node in self._get_predecessors(current):
                    self._update_node(p_node)

    def update_and_replan(self, new_start: Tuple, cost_updates: Dict):
        self.km += self.heuristic(self.start)
        self.start = new_start
        
        for cell, cost in cost_updates.items():
            self.cost_map[cell] = cost
            # When a cost changes, we only need to update the node itself initially.
            # The compute_shortest_path loop will propagate the changes.
            self._update_node(cell)

        self.compute_shortest_path()

    def get_path(self) -> Optional[List[Tuple]]:
        if self.g_score[self.start] == float('inf'):
            logging.warning("D* Lite: No path found, g_score for start is infinity.")
            return None
        path, current = [self.start], self.start
        while current != self.goal:
            if len(path) > 5000: # Safety break to prevent infinite loops in path reconstruction
                logging.error("D* Lite path reconstruction exceeded max length.")
                return None
            
            successors = list(self._get_successors(current))
            if not successors:
                logging.error(f"D* Lite path reconstruction failed: no successors for {current}")
                return None
            
            # Find the next best step by looking at the cost-to-go (g_score) of successors
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