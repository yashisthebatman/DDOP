import pytest
import numpy as np
from utils.a_star import AStarPlanner

@pytest.fixture
def planner():
    return AStarPlanner()

def test_a_star_finds_path_in_open_grid(planner):
    grid = np.full((10, 10, 10), True)
    start, goal = (1, 1, 1), (8, 8, 8)
    path = planner.find_path(grid, start, goal)
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    assert len(path) > 2

def test_a_star_returns_none_if_no_path(planner):
    grid = np.full((10, 10, 10), True)
    start, goal = (1, 1, 1), (8, 8, 8)
    grid[7:10, 7:10, 7:10] = False
    grid[8, 8, 8] = True
    path = planner.find_path(grid, start, goal)
    assert path is None

def test_a_star_navigates_simple_obstacle(planner):
    grid = np.full((10, 10, 10), True)
    start, goal = (1, 5, 5), (8, 5, 5)
    grid[5, 3:8, :] = False
    path = planner.find_path(grid, start, goal)
    assert path is not None
    for point in path:
        assert grid[point[0], point[1], point[2]] == True

def test_a_star_shortest_path_property(planner):
    grid = np.full((20, 20, 1), True)
    start, goal = (2, 10, 0), (18, 10, 0)
    grid[5:15, 9, 0] = False
    grid[5:15, 11, 0] = False
    grid[5, 9:12, 0] = False
    path = planner.find_path(grid, start, goal)
    assert path is not None
    goes_up = any(p[1] > start[1] for p in path)
    assert goes_up is True
    goes_down = any(p[1] < start[1] for p in path)
    assert goes_down is False