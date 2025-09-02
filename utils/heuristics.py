import numpy as np
from typing import Tuple, Callable, TYPE_CHECKING
from config import A_STAR_HEURISTIC_WEIGHT, DRONE_SPEED_MPS

if TYPE_CHECKING:
    from utils.coordinate_manager import CoordinateManager

WorldCoord = Tuple[float, float, float]
GridCoord = Tuple[int, int, int]

class HeuristicProvider:
    """
    Provides heuristic functions for different pathfinding contexts and goals.
    This decouples the heuristic logic from the main planner.
    """
    def __init__(self, coord_manager: 'CoordinateManager'):
        self.coord_manager = coord_manager

    def get_heuristic(
        self, mode: str, goal: WorldCoord, payload_kg: float, time_weight: float = 0.5
    ) -> Callable[[WorldCoord, WorldCoord], float]:
        """
        Returns a heuristic function for the abstract graph (world coordinates).
        The returned function MUST accept two arguments (u, v) to be compatible with NetworkX.
        """
        goal_np = np.array(goal)

        # FIX: The heuristic function for NetworkX A* must accept two arguments: u (current) and v (goal).
        def time_heuristic(u: WorldCoord, v: WorldCoord) -> float:
            """Estimates time cost as distance / speed."""
            dist = np.linalg.norm(np.array(u) - goal_np)
            return (dist / DRONE_SPEED_MPS) * A_STAR_HEURISTIC_WEIGHT

        # FIX: The heuristic function signature is updated to (u, v).
        def energy_heuristic(u: WorldCoord, v: WorldCoord) -> float:
            """Estimates energy cost. Simplified to be proportional to distance."""
            dist = np.linalg.norm(np.array(u) - goal_np)
            return dist * A_STAR_HEURISTIC_WEIGHT

        if mode == "time":
            return time_heuristic
        elif mode == "energy":
            return energy_heuristic
        else: # balanced
            # FIX: The heuristic function signature is updated to (u, v).
            def balanced_heuristic(u: WorldCoord, v: WorldCoord) -> float:
                time_cost = time_heuristic(u, v)
                energy_cost = energy_heuristic(u, v)
                return (time_weight * time_cost + (1 - time_weight) * energy_cost)
            return balanced_heuristic

    def get_grid_heuristic(self, goal: GridCoord) -> Callable[[GridCoord], float]:
        """
        Returns a simple Euclidean distance heuristic for grid-based algorithms like D* Lite.
        This heuristic correctly takes only one argument (the current node).
        """
        goal_np = np.array(goal)
        def heuristic(node: GridCoord) -> float:
            return np.linalg.norm(np.array(node) - goal_np)
        return heuristic