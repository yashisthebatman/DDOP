# ==============================================================================
# File: utils/a_star.py (NEW FILE - FINALIZED)
# ==============================================================================
"""A standard, robust A* pathfinding implementation to serve as a guaranteed fallback."""
import heapq
import numpy as np
from itertools import product

class AStarSearch:
    def __init__(self, start, goal, is_obstructed_func, heuristic, grid_shape):
        self.start, self.goal = start, goal
        self.is_obstructed, self.heuristic, self.grid_shape = is_obstructed_func, heuristic, grid_shape
        self.open_set, self.came_from, self.g_score = [], {}, {start: 0}
        self.directions = list(product([-1, 0, 1], repeat=3)); self.directions.remove((0, 0, 0))

    def search(self):
        """Performs the A* search."""
        heapq.heappush(self.open_set, (self.heuristic.calculate(self.start), self.start))
        while self.open_set:
            _, current = heapq.heappop(self.open_set)
            if current == self.goal: return self._reconstruct_path(current)
            for direction in self.directions:
                neighbor = tuple(np.array(current) + np.array(direction))
                if not self._is_valid_position(neighbor) or self.is_obstructed(neighbor): continue
                move_cost = np.linalg.norm(np.array(direction))
                tentative_g_score = self.g_score[current] + move_cost
                if tentative_g_score < self.g_score.get(neighbor, float('inf')):
                    self.came_from[neighbor], self.g_score[neighbor] = current, tentative_g_score
                    f_score = tentative_g_score + self.heuristic.calculate(neighbor)
                    heapq.heappush(self.open_set, (f_score, neighbor))
        return None

    def _is_valid_position(self, pos):
        """Checks if a position is within the grid boundaries."""
        x, y, z = pos
        return (0 <= x < self.grid_shape[0] and 0 <= y < self.grid_shape[1] and 0 <= z < self.grid_shape[2])

    def _reconstruct_path(self, current):
        """Builds the path from the goal back to the start."""
        path = [current]
        while current in self.came_from: current = self.came_from[current]; path.append(current)
        path.reverse(); return path