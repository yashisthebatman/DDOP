import pytest
from utils.jump_point_search import JumpPointSearch
from utils.coordinate_manager import CoordinateManager
import numpy as np

# Mock Heuristic class for testing purposes
class MockHeuristic:
    def calculate(self, node):
        return 0

# Fixture to set up the JPS environment for each test
@pytest.fixture
def jps_setup():
    manager = CoordinateManager(grid_depth=10)
    obstacles = set()
    def is_obstructed(pos):
        return pos in obstacles
    
    jps = JumpPointSearch(
        start=(0, 0, 0), 
        goal=(9, 9, 9), 
        is_obstructed_func=is_obstructed,
        heuristic=MockHeuristic(),
        coord_manager=manager
    )
    return jps, obstacles

# Test to ensure path reconstruction works as expected
def test_jps_path_reconstruction(jps_setup):
    jps, _ = jps_setup
    jps.came_from = {(1, 1, 1): (0, 0, 0), (2, 2, 2): (1, 1, 1)}
    path = jps._reconstruct_path_grid((2, 2, 2))
    assert path == [(0, 0, 0), (1, 1, 1), (2, 2, 2)]

# Test JPS on a simple straight path with no obstacles
def test_jps_straight_path(jps_setup):
    jps, _ = jps_setup
    jps.goal = (5, 0, 0)
    path = jps.search()
    assert path is not None
    assert path == [(i, 0, 0) for i in range(6)]

# Test JPS behavior with a simple diagonal path
def test_jps_diagonal_path(jps_setup):
    jps, _ = jps_setup
    jps.goal = (3, 3, 0)
    path = jps.search()
    assert path is not None
    assert path == [(i, i, 0) for i in range(4)]

# Test scenario where no path is possible due to obstacles
def test_jps_no_path(jps_setup):
    jps, obstacles = jps_setup
    jps.goal = (2, 0, 0)
    # FIX: Create a proper 3D wall to block the path.
    # The algorithm was correctly finding a path around the single obstacle.
    for y in range(-1, 2):
        for z in range(-1, 2):
            obstacles.add((1, y, z))
    path = jps.search()
    # JPS should now correctly return None as the path is fully blocked
    assert path is None

# Test handling of the start and goal being the same point
def test_jps_start_is_goal(jps_setup):
    jps, _ = jps_setup
    jps.goal = jps.start
    path = jps.search()
    assert path == [jps.start]

# Test for forced neighbors in the XY plane
def test_jps_forced_neighbor_xy(jps_setup):
    jps, obstacles = jps_setup
    obstacles.add((1, 2, 0))
    jps.start = (1, 1, 0)
    jps.goal = (3, 3, 0)
    assert jps._has_forced_neighbor(node=(2, 2, 0), direction=(1, 1, 0))

# Test for forced neighbors in the XZ plane
def test_jps_forced_neighbor_xz(jps_setup):
    jps, obstacles = jps_setup
    obstacles.add((1, 0, 2))
    jps.start = (1, 0, 1)
    jps.goal = (3, 0, 3)
    assert jps._has_forced_neighbor(node=(2, 0, 2), direction=(1, 0, 1))

# Test for forced neighbors in the YZ plane
def test_jps_forced_neighbor_yz(jps_setup):
    jps, obstacles = jps_setup
    obstacles.add((0, 1, 2))
    jps.start = (0, 1, 1)
    jps.goal = (0, 3, 3)
    assert jps._has_forced_neighbor(node=(0, 2, 2), direction=(0, 1, 1))

# Test a more complex path that requires navigating around an obstacle
def test_jps_path_around_obstacle(jps_setup):
    jps, obstacles = jps_setup
    jps.goal = (4, 0, 0)
    # Create a wall of obstacles
    for i in range(4):
        obstacles.add((2, i, 0))
    path = jps.search()
    assert path is not None
    assert (2, 0, 0) not in path

# Test the internal _get_line_cells function for path interpolation
def test_get_line_cells(jps_setup):
    jps, _ = jps_setup
    path_segment = jps._get_line_cells((0, 0, 0), (3, 3, 3))
    assert path_segment == [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]