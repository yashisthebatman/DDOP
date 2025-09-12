# FILE: utils/a_star.py
import heapq
from typing import List, Optional, Tuple, Set
import numpy as np

GridPosition = Tuple[int, int, int]

class AStarPlanner:
    def find_path(self, grid: np.ndarray, start_grid: GridPosition, goal_grid: GridPosition) -> Optional[List[GridPosition]]:
        open_set = [(0, start_grid)]
        came_from: dict[GridPosition, Optional[GridPosition]] = {start_grid: None}
        g_score: dict[GridPosition, float] = {start_grid: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_grid:
                return self._reconstruct_path(came_from, current)

            for neighbor, cost in self._get_neighbors_with_cost(current, grid):
                tentative_g_score = g_score[current] + cost
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self._heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score, neighbor))
        
        return None

    def _get_neighbors_with_cost(self, pos: GridPosition, grid: np.ndarray) -> List[Tuple[GridPosition, float]]:
        """
        Gets all valid neighbors with their movement cost.
        Straight moves cost 1, 2D diagonal sqrt(2), 3D diagonal sqrt(3).
        """
        neighbors = []
        x, y, z = pos
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 0 <= nz < grid.shape[2]:
                        if grid[nx, ny, nz]:
                            # Calculate cost based on distance
                            dist = np.sqrt(dx**2 + dy**2 + dz**2)
                            # FIX: Add a tiny cost incentive for moving "up" (positive y) to break ties deterministically.
                            if dy > 0:
                                dist -= 1e-6
                            neighbors.append(((nx, ny, nz), dist))
        return neighbors

    def _heuristic(self, a: GridPosition, b: GridPosition) -> float:
        return np.linalg.norm(np.array(a) - np.array(b))

    def _reconstruct_path(self, came_from: dict, current: GridPosition) -> List[GridPosition]:
        path = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        return path[::-1]