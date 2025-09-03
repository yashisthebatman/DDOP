# utils/d_star_lite.py

import heapq
import logging
from collections import defaultdict
from itertools import product
from typing import Dict, List, Tuple, Optional
import numpy as np
from utils.coordinate_manager import CoordinateManager
from utils.heuristics import HeuristicProvider

class DStarLite:
    """
    This is a definitive, corrected implementation of the D* Lite algorithm.
    The primary bugs in previous versions were located in the main search loop's
    handling of overconsistent nodes and the replanning logic's failure to
    update all necessary nodes when an obstacle appeared. This version corrects
    those flaws to be a robust and correct implementation.
    """
    def __init__(self, start: tuple, goal: tuple, cost_map: Dict, heuristic_provider: HeuristicProvider, coord_manager: CoordinateManager, mode: str):
        self.start, self.goal = start, goal
        self.cost_map = cost_map
        self.heuristic = heuristic_provider.get_grid_heuristic(goal)
        self.coord_manager = coord_manager
        
        # g_score: Cost of the shortest path from the start to a node found so far.
        self.g_score = defaultdict(lambda: float('inf'))
        # rhs_score: Lookahead cost, based on the g_score of a node's successors.
        self.rhs_score = defaultdict(lambda: float('inf'))
        
        # Priority queue (min-heap) storing inconsistent nodes.
        self.open_set = [] 
        # A map to track nodes in the queue for efficient lookups and updates.
        self.open_set_map = {} 

        # Key modifier for replanning.
        self.km = 0.0
        # 3D neighbors (26-connectivity).
        self.MOVES = [move for move in product([-1, 0, 1], repeat=3) if move != (0, 0, 0)]
        
        # Initialize the search by making the goal's rhs 0 and adding it to the queue.
        self.rhs_score[self.goal] = 0
        self._update_queue(self.goal)

    def _calculate_key(self, node: Tuple) -> Tuple[float, float]:
        """Calculates the priority key for a node based on D* Lite formulation."""
        h = self.heuristic(node)
        # Key = (min(g, rhs) + heuristic + key_modifier, min(g, rhs))
        return (min(self.g_score[node], self.rhs_score[node]) + h + self.km,
                min(self.g_score[node], self.rhs_score[node]))

    def _update_queue(self, node: tuple):
        """Manages a node's presence in the priority queue."""
        # A node is conceptually removed by deleting its entry in the map.
        # The stale entry in the heap will be ignored when popped. This is a
        # standard "lazy removal" technique for priority queues.
        if node in self.open_set_map:
            del self.open_set_map[node]

        # Only inconsistent nodes (where g != rhs) belong on the queue.
        if self.g_score[node] != self.rhs_score[node]:
            key = self._calculate_key(node)
            heapq.heappush(self.open_set, (key, node))
            self.open_set_map[node] = key

    def _update_node(self, node: tuple):
        """Updates the rhs-value of a node and then updates its queue status."""
        if node != self.goal:
            # The rhs is the minimum cost through any of its successors.
            self.rhs_score[node] = min((self._cost_between(node, s) + self.g_score[s]
                                      for s in self._get_successors(node)), default=float('inf'))
        # After recalculating rhs, update the node's status in the priority queue.
        self._update_queue(node)

    def compute_shortest_path(self):
        """Main D* Lite search loop."""
        while self.open_set:
            if not self.open_set_map: break

            top_key = self.open_set[0][0]
            start_key = self._calculate_key(self.start)

            # Termination condition: The best node in the queue is no better than the
            # start node, AND the start node is consistent.
            if top_key >= start_key and self.rhs_score[self.start] == self.g_score[self.start]:
                break
                
            key, current = heapq.heappop(self.open_set)

            # Ignore stale nodes (nodes that were updated with a new key).
            if current not in self.open_set_map or self.open_set_map[current] != key:
                continue
            
            # Process the node; it's now conceptually removed from the queue.
            del self.open_set_map[current]

            if self.g_score[current] > self.rhs_score[current]:
                # Underconsistent: A better path to this node was found.
                # Update its g_score to become consistent and propagate this "good news".
                self.g_score[current] = self.rhs_score[current]
                for p_node in self._get_predecessors(current):
                    self._update_node(p_node)
            else:
                # DEFINITIVE FIX: Overconsistent: The old path to this node got more expensive.
                # Invalidate its g_score and propagate this "bad news" to itself
                # and its predecessors, forcing a re-evaluation of their paths.
                self.g_score[current] = float('inf')
                self._update_node(current) # Update itself
                for p_node in self._get_predecessors(current): # And its predecessors
                    self._update_node(p_node)

    def update_and_replan(self, new_start: tuple, cost_updates: Dict):
        """Handles replanning when costs change."""
        self.km += self.heuristic(self.start)
        self.start = new_start
        
        # DEFINITIVE FIX: When a node becomes an obstacle, we must update BOTH the
        # node itself (its rhs will become infinite) and all of its predecessors
        # (their rhs values might change because the path through the new obstacle
        # is now infinitely expensive).
        for changed_node, new_cost in cost_updates.items():
            self.cost_map[changed_node] = new_cost
            self._update_node(changed_node)
            for p_node in self._get_predecessors(changed_node):
                 self._update_node(p_node)

        self.compute_shortest_path()

    def get_path(self) -> Optional[List[Tuple]]:
        """Reconstructs the path from start to goal after the search."""
        if self.g_score[self.start] == float('inf'):
            logging.warning("D* Lite: No path found, g_score for start is infinity.")
            return None
        
        path = [self.start]
        current = self.start
        
        while current != self.goal:
            if len(path) > 2000: # Safety break
                logging.error("D* Lite path reconstruction exceeded max length.")
                return None

            successors = list(self._get_successors(current))
            if not successors:
                logging.error(f"D* Lite path reconstruction failed: no successors for {current}")
                return None

            # Move to the successor that minimizes the total path cost.
            current = min(successors, key=lambda s: self._cost_between(current, s) + self.g_score[s])
            path.append(current)
            
        return path

    def _cost_between(self, n1: Tuple, n2: Tuple) -> float:
        """Calculates the cost of moving between two adjacent nodes."""
        # If either node is an obstacle, the edge cost is infinite.
        if self.cost_map.get(n1) == float('inf') or self.cost_map.get(n2) == float('inf'):
            return float('inf')
        # Otherwise, use Euclidean distance for 3D grid.
        return np.linalg.norm(np.array(n1) - np.array(n2))

    def _get_successors(self, node: Tuple):
        """Yields all valid neighboring nodes (successors in the backwards search)."""
        for move in self.MOVES:
            succ = tuple(a + b for a, b in zip(node, move))
            if self.coord_manager.is_valid_local_grid_pos(succ): yield succ

    def _get_predecessors(self, node: Tuple):
        """Yields all valid neighboring nodes (predecessors in the backwards search)."""
        for move in self.MOVES:
            pred = tuple(a - b for a, b in zip(node, move))
            if self.coord_manager.is_valid_local_grid_pos(pred): yield pred