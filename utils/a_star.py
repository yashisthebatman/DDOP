# ==============================================================================
# File: utils/a_star.py (NEW FILE)
# ==============================================================================
"""A standard, robust A* pathfinding implementation to serve as a guaranteed fallback."""

import heapq
import numpy as np
from itertools import product

class AStarSearch:
    def __init__(self, start, goal, is_obstructed_func, heuristic, grid_shape):
        self.start = start
        self.goal = goal
        self.is_obstructed = is_obstructed_func
        self.heuristic = heuristic
        self.grid_shape = grid_shape
        
        self.open_set = []
        self.came_from = {}
        self.g_score = {start: 0}
        
        # Define all 26 possible 3D moves (cardinal, planar, and full diagonal)
        self.directions = list(product([-1, 0, 1], repeat=3))
        self.directions.remove((0, 0, 0))

    def search(self):
        """Performs the A* search."""
        heapq.heappush(self.open_set, (self.heuristic.calculate(self.start), self.start))

        while self.open_set:
            _, current = heapq.heappop(self.open_set)

            if current == self.goal:
                return self._reconstruct_path(current)

            for direction in self.directions:
                neighbor = (current[0] + direction[0], 
                            current[1] + direction[1], 
                            current[2] + direction[2])

                if not self._is_valid_position(neighbor) or self.is_obstructed(neighbor):
                    continue

                # The cost to move from current to neighbor is the Euclidean distance
                move_cost = np.linalg.norm(np.array(direction))
                tentative_g_score = self.g_score[current] + move_cost

                if tentative_g_score < self.g_score.get(neighbor, float('inf')):
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic.calculate(neighbor)
                    heapq.heappush(self.open_set, (f_score, neighbor))
        
        return None # No path found

    def _is_valid_position(self, position):
        """Checks if a position is within the grid boundaries."""
        x, y, z = position
        return (0 <= x < self.grid_shape[0] and
                0 <= y < self.grid_shape[1] and
                0 <= z < self.grid_shape[2])

    def _reconstruct_path(self, current):
        """Builds the path from the goal back to the start."""
        path = [current]
        while current in self.came_from:
            current = self.came_from[current]
            path.append(current)
        path.reverse()
        return path