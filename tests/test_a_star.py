import pytest
import numpy as np
from utils.a_star import AStarPlanner

@pytest.fixture
def planner():
    return AStarPlanner()

def test_a_star_finds_path_in_open_grid(planner):
    grid = np.full((10, 10, 10), 1.0)
    start, goal = (1, 1, 1), (8, 8, 8)
    path = planner.find_path(grid, start, goal)
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    assert len(path) > 2

def test_a_star_returns_none_if_no_path(planner):
    grid = np.full((10, 10, 10), 1.0)
    start, goal = (1, 1, 1), (8, 8, 8)
    grid[4, :, :] = np.inf # Wall
    path = planner.find_path(grid, start, goal)
    assert path is None

def test_a_star_navigates_simple_obstacle(planner):
    grid = np.full((10, 10, 10), 1.0)
    start, goal = (1, 5, 5), (8, 5, 5)
    grid[5, 3:8, :] = np.inf
    path = planner.find_path(grid, start, goal)
    assert path is not None
    for point in path:
        assert not np.isinf(grid[point[0], point[1], point[2]])

def test_a_star_chooses_cheapest_path_not_shortest(planner):
    """
    Tests that A* correctly chooses a longer but cheaper path over a shorter but more expensive one.
    """
    grid = np.full((20, 20, 1), 10.0) # Base cost is high
    start, goal = (2, 10, 0), (18, 10, 0)

    # Create a very cheap "highway" that requires a detour.
    grid[5:15, 5, 0] = 1.0 # The "down" path is now much cheaper
    
    # Make the direct path possible but very expensive
    grid[start[0]+1:goal[0]-1, start[1], 0] = 100.0

    path = planner.find_path(grid, start, goal)
    assert path is not None
    
    # The path MUST go down to y=5 to use the cheap highway.
    goes_down_to_highway = any(p[1] == 5 for p in path)
    assert goes_down_to_highway is True