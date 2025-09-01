# utils/lpa_star.py
import heapq
import numpy as np
from collections import defaultdict

class LPAStar:
    """
    A stateful implementation of the Lifelong Planning A* (LPA*) algorithm.
    This class maintains its state between planning and replanning calls,
    making it extremely efficient for dynamic environments where costs change.
    """

    def __init__(self, grid: np.ndarray, start: tuple, goal: tuple, heuristic: 'Heuristic'):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.heuristic = heuristic

        # LPA* specific data structures
        self.g_score = defaultdict(lambda: float('inf'))
        self.rhs_score = defaultdict(lambda: float('inf'))
        
        # Priority queue (U)
        # We store (key, node) where key is a tuple (k1, k2)
        self.open_set = []
        self.open_set_map = {} # For quick lookups and updates

        # Initialize
        self.rhs_score[self.start] = 0
        key = self._calculate_key(self.start)
        heapq.heappush(self.open_set, (key, self.start))
        self.open_set_map[self.start] = key

    def _calculate_key(self, node: tuple) -> tuple:
        """Calculates the two-part priority key for a node."""
        h = self.heuristic.calculate(node)
        g = self.g_score[node]
        rhs = self.rhs_score[node]
        min_score = min(g, rhs)
        return (min_score + h, min_score)

    def _update_node(self, node: tuple):
        """
        The core of LPA*. Updates a node's rhs-value and its position in the
        priority queue if it has become locally inconsistent.
        """
        # 1. Update rhs-value for the node (unless it's the start)
        if node != self.start:
            min_rhs = float('inf')
            # Look at predecessors (neighbors that can lead to this node)
            for p_node in self._get_neighbors(node):
                cost = self.heuristic.cost_between(p_node, node, None) # p_prev is ignored for cost
                min_rhs = min(min_rhs, self.g_score[p_node] + cost)
            self.rhs_score[node] = min_rhs

        # 2. Remove from priority queue if it's already there
        if node in self.open_set_map:
            # A simple removal is difficult with heapq. We mark it as removed.
            # A cleaner but slower way is to rebuild the heap. For performance,
            # we will handle duplicates upon popping. Here, we just remove from map.
            del self.open_set_map[node]

        # 3. If node is locally inconsistent, add it back to the priority queue
        if self.g_score[node] != self.rhs_score[node]:
            key = self._calculate_key(node)
            heapq.heappush(self.open_set, (key, node))
            self.open_set_map[node] = key

    def compute_shortest_path(self):
        """
        Runs the main LPA* loop, processing the priority queue until the goal is
        consistent or the best path to it is found.
        """
        while self.open_set:
            # Check stopping condition
            goal_key = self._calculate_key(self.goal)
            top_key, current = self.open_set[0]
            if top_key >= goal_key and self.g_score[self.goal] == self.rhs_score[self.goal]:
                break

            # Pop node with the smallest key
            key, current = heapq.heappop(self.open_set)
            
            # If a node is in the queue multiple times, we only process the best one
            if current not in self.open_set_map or self.open_set_map[current] < key:
                continue
            del self.open_set_map[current]

            # Process the popped node
            if self.g_score[current] > self.rhs_score[current]:
                # Overconsistent node: Update g-score and propagate changes to successors
                self.g_score[current] = self.rhs_score[current]
                for s_node in self._get_neighbors(current):
                    self._update_node(s_node)
            else:
                # Underconsistent node: Update g-score and propagate changes
                self.g_score[current] = float('inf')
                self._update_node(current) # Update self first
                for s_node in self._get_neighbors(current):
                    self._update_node(s_node)

    def get_path(self) -> list or None:
        """Reconstructs the path from goal to start using g-scores."""
        if self.g_score[self.goal] == float('inf'):
            return None  # No path exists

        path = [self.goal]
        current = self.goal
        while current != self.start:
            neighbors = self._get_neighbors(current)
            best_neighbor = min(
                neighbors,
                key=lambda n: self.g_score[n] + self.heuristic.cost_between(n, current, None)
            )
            current = best_neighbor
            path.append(current)
        
        path.append(self.start)
        return path[::-1]

    def _get_neighbors(self, node: tuple) -> list:
        """Returns valid, non-obstacle neighbors of a node."""
        neighbors = []
        moves = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1] if not (dx == 0 and dy == 0 and dy == 0)]
        for move in moves:
            neighbor = (node[0] + move[0], node[1] + move[1], node[2] + move[2])
            if (0 <= neighbor[0] < self.grid.shape[0] and
                0 <= neighbor[1] < self.grid.shape[1] and
                0 <= neighbor[2] < self.grid.shape[2] and
                self.grid[neighbor] == 0):
                neighbors.append(neighbor)
        return neighbors