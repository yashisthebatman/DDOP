import numpy as np
from typing import Tuple, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.coordinate_manager import CoordinateManager

GridCoord = Tuple[int, int, int]

class HeuristicProvider:
    """
    Provides heuristic functions for grid-based pathfinding algorithms like D* Lite.
    """
    def __init__(self, coord_manager: 'CoordinateManager'):
        self.coord_manager = coord_manager

    def get_grid_heuristic(self, goal: GridCoord) -> Callable[[GridCoord], float]:
        """
        Returns a simple Euclidean distance heuristic for grid-based algorithms.
        """
        goal_np = np.array(goal)
        def heuristic(node: GridCoord) -> float:
            return np.linalg.norm(np.array(node) - goal_np)
        return heuristic