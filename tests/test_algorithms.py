import pytest
from unittest.mock import MagicMock
import numpy as np

# This file now exclusively tests the D* Lite tactical replanning algorithm,
# as A* and JPS have been removed from the production architecture.
from utils.d_star_lite import DStarLite
from utils.heuristics import HeuristicProvider
from utils.coordinate_manager import CoordinateManager

@pytest.fixture
def grid_world_3d():
    """Provides a consistent 3D grid environment for testing."""
    coord_manager = MagicMock(spec=CoordinateManager)
    coord_manager.is_valid_local_grid_pos.side_effect = lambda pos: (
        0 <= pos[0] < 10 and 0 <= pos[1] < 10 and 0 <= pos[2] < 10
    )
    heuristic_provider = HeuristicProvider(coord_manager)
    # A solid "wall" of obstacles where x=5
    obstacles = {(5, y, z) for y in range(10) for z in range(10)}
    cost_map = {obs: float('inf') for obs in obstacles}

    return {
        "coord_manager": coord_manager,
        "heuristic_provider": heuristic_provider,
        "cost_map": cost_map
    }

def test_d_star_lite_initial_path(grid_world_3d):
    """Tests if D* Lite can find a path around the initial obstacle wall."""
    start, goal = (1, 5, 5), (8, 5, 5)
    d_star = DStarLite(
        start, goal,
        cost_map=grid_world_3d["cost_map"],
        heuristic_provider=grid_world_3d["heuristic_provider"],
        coord_manager=grid_world_3d["coord_manager"],
        mode='time'
    )
    d_star.compute_shortest_path()
    path = d_star.get_path()
    assert path is not None, "D* Lite should have found a path."
    assert all(p[0] != 5 for p in path), "Path should navigate around the wall at x=5."

def test_d_star_lite_replan(grid_world_3d):
    """Tests if D* Lite can successfully replan when a new obstacle appears."""
    start, goal = (1, 5, 5), (8, 5, 5)
    d_star = DStarLite(
        start, goal,
        cost_map=grid_world_3d["cost_map"].copy(), # Use a copy for modification
        heuristic_provider=grid_world_3d["heuristic_provider"],
        coord_manager=grid_world_3d["coord_manager"],
        mode='time'
    )
    # First, compute an initial path
    d_star.compute_shortest_path()
    
    # Introduce a new obstacle on the expected path and move the drone
    cost_updates = {(3, 5, 5): float('inf')}
    d_star.update_and_replan(new_start=(2, 5, 5), cost_updates=cost_updates)
    replan_path = d_star.get_path()

    assert replan_path is not None, "D* Lite should have found a replanned path."
    assert (3, 5, 5) not in replan_path, "Replanned path should avoid the new obstacle."
    assert all(p[0] != 5 for p in replan_path), "Replanned path should still avoid the original wall."