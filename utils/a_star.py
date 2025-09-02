import heapq
import numpy as np
from itertools import product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.coordinate_manager import CoordinateManager

class AStarSearch:
    def __init__(self, start: tuple, goal: tuple, is_obstructed_func: callable, heuristic, coord_manager: 'CoordinateManager'):
        self.start = start
        self.goal = goal
        self.is_obstructed = is_obstructed_func
        self.heuristic = heuristic
        self.coord_manager = coord_manager
        self.open_set = []
        self.came_from = {}
        self.g_score = {start: 0}
        self.directions = list(product([-1, 0, 1], repeat=3))
        self.directions.remove((0, 0, 0))

    def search(self):
        # FIX: Add a guard clause to fail fast if the start or goal is impossible.
        # This prevents the algorithm from running unnecessarily and makes it more robust.
        if self.is_obstructed(self.start) or self.is_obstructed(self.goal):
            return None
            
        heapq.heappush(self.open_set, (self.heuristic(self.start), self.start))
        while self.open_set:
            _, current = heapq.heappop(self.open_set)
            if current == self.goal:
                return self._reconstruct_path(current)
            for direction in self.directions:
                neighbor = (current[0] + direction[0], current[1] + direction[1], current[2] + direction[2])
                if not self.coord_manager.is_valid_local_grid_pos(neighbor) or self.is_obstructed(neighbor):
                    continue
                move_cost = np.linalg.norm(np.array(direction))
                tentative_g_score = self.g_score[current] + move_cost
                if tentative_g_score < self.g_score.get(neighbor, float('inf')):
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor)
                    heapq.heappush(self.open_set, (f_score, neighbor))
        return None

    def _reconstruct_path(self, current):
        path = [current]
        while current in self.came_from:
            # FIX: The previous line-of-sight check was flawed and has been removed.
            # The main search loop's check on neighbors is the correct place for validation.
            current = self.came_from[current]
            path.append(current)
        path.reverse()
        return path