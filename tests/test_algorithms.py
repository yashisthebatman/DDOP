import pytest
from unittest.mock import MagicMock
import numpy as np

from utils.a_star import AStarSearch
from utils.d_star_lite import DStarLite
from utils.jump_point_search import JumpPointSearch
from utils.coordinate_manager import CoordinateManager
from utils.heuristics import HeuristicProvider

@pytest.fixture
def grid_world_3d():
    coord_manager = MagicMock(spec=CoordinateManager)
    coord_manager.is_valid_local_grid_pos.side_effect = lambda pos: (
        0 <= pos[0] < 10 and 0 <= pos[1] < 10 and 0 <= pos[2] < 10
    )
    heuristic_provider = HeuristicProvider(coord_manager)
    obstacles = {(5, y, z) for y in range(10) for z in range(10)}
    is_obstructed_func = lambda pos: pos in obstacles
    
    # FIX: Create a cost_map for D* Lite tests from the obstacle set.
    cost_map = {obs: float('inf') for obs in obstacles}

    return {
        "coord_manager": coord_manager,
        "is_obstructed": is_obstructed_func,
        "heuristic_provider": heuristic_provider,
        "cost_map": cost_map
    }

def test_a_star_simple_path(grid_world_3d):
    start, goal = (1, 1, 1), (4, 1, 1)
    heuristic = grid_world_3d["heuristic_provider"].get_grid_heuristic(goal)
    a_star = AStarSearch(
        start, goal, is_obstructed_func=lambda pos: False,
        heuristic=heuristic, coord_manager=grid_world_3d["coord_manager"]
    )
    path = a_star.search()
    assert path is not None and path[0] == start and path[-1] == goal

def test_a_star_path_around_wall(grid_world_3d):
    start, goal = (1, 5, 5), (8, 5, 5)
    heuristic = grid_world_3d["heuristic_provider"].get_grid_heuristic(goal)
    a_star = AStarSearch(
        start, goal, is_obstructed_func=grid_world_3d["is_obstructed"],
        heuristic=heuristic, coord_manager=grid_world_3d["coord_manager"]
    )
    path = a_star.search()
    assert path is not None and all(p[0] != 5 for p in path)

def test_a_star_no_path(grid_world_3d):
    start, goal = (1, 5, 5), (5, 5, 5)
    heuristic = grid_world_3d["heuristic_provider"].get_grid_heuristic(goal)
    a_star = AStarSearch(
        start, goal, is_obstructed_func=grid_world_3d["is_obstructed"],
        heuristic=heuristic, coord_manager=grid_world_3d["coord_manager"]
    )
    assert a_star.search() is None

def test_d_star_lite_initial_path(grid_world_3d):
    start, goal = (1, 5, 5), (8, 5, 5)
    d_star = DStarLite(
        start, goal,
        # FIX: Pass the cost_map with the wall obstacle.
        cost_map=grid_world_3d["cost_map"],
        heuristic_provider=grid_world_3d["heuristic_provider"],
        coord_manager=grid_world_3d["coord_manager"],
        mode='time'
    )
    d_star.compute_shortest_path()
    path = d_star.get_path()
    assert path is not None and all(p[0] != 5 for p in path)

def test_d_star_lite_replan(grid_world_3d):
    start, goal = (1, 5, 5), (8, 5, 5)
    d_star = DStarLite(
        start, goal,
        # FIX: Pass the initial wall obstacle.
        cost_map=grid_world_3d["cost_map"].copy(),
        heuristic_provider=grid_world_3d["heuristic_provider"],
        coord_manager=grid_world_3d["coord_manager"],
        mode='time'
    )
    d_star.compute_shortest_path()
    # Add a new obstacle
    cost_updates = {(3, 5, 5): float('inf')}
    d_star.update_and_replan(new_start=(2, 5, 5), cost_updates=cost_updates)
    replan_path = d_star.get_path()
    assert replan_path is not None
    assert (3, 5, 5) not in replan_path
    assert all(p[0] != 5 for p in replan_path)

# JPS is not used in the final planner, but its test is kept for completeness
def test_jps_simple_path(grid_world_3d):
    start, goal = (1, 1, 1), (4, 4, 4)
    heuristic = grid_world_3d["heuristic_provider"].get_grid_heuristic(goal)
    jps = JumpPointSearch(
        start, goal, is_obstructed_func=lambda pos: False,
        heuristic=heuristic, coord_manager=grid_world_3d["coord_manager"]
    )
    path = jps.search()
    assert path is not None and path[0] == start and path[-1] == goal