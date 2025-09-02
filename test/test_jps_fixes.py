# File: tests/test_jps_fixes.py
import pytest
from utils.jump_point_search import JumpPointSearch
from utils.coordinate_manager import CoordinateManager

class MockHeuristic:
    def calculate(self, node):
        return 0

@pytest.fixture
def jps_setup():
    manager = CoordinateManager(grid_depth=10)
    # Create a mock is_obstructed function that uses a set of obstacle points
    obstacles = set()
    def is_obstructed(pos):
        return pos in obstacles
    
    jps = JumpPointSearch(
        start=(0,0,0), 
        goal=(9,9,9), 
        is_obstructed_func=is_obstructed,
        heuristic=MockHeuristic(),
        coord_manager=manager
    )
    return jps, obstacles

def test_jps_path_reconstruction(jps_setup):
    jps, _ = jps_setup
    jps.came_from = {(2, 2, 2): (0, 0, 0)}
    path = jps._reconstruct_path_grid((2, 2, 2))
    assert path == [(0, 0, 0), (1, 1, 1), (2, 2, 2)]

def test_jps_straight_path(jps_setup):
    jps, _ = jps_setup
    jps.goal = (5, 0, 0)
    path = jps.search()
    assert path is not None
    assert path == [(i, 0, 0) for i in range(6)]

def test_jps_forced_neighbor_xy(jps_setup):
    jps, obstacles = jps_setup
    jps.start = (1, 1, 1)
    jps.goal = (3, 3, 1)
    
    # Create an obstacle that should force a new jump point
    # Move from (1,1) to (2,2) should be fine, but from (2,2) to (3,3) should
    # find a forced neighbor at (3,2) because (2,2) is blocked.
    obstacles.add((2, 2, 1))
    
    # This is a complex test, we check the internal _has_forced_neighbor method
    # Direction is (1, 1, 0) moving towards (3,2,1)
    # The obstacle is at (2,2,1). The node being checked is (3,2,1).
    # The parent of (3,2,1) would be (2,1,1) if it came from that way.
    # A better test:
    # We are at node (2,1,1), moving with direction (1,1,0)
    # Next node is (3,2,1). Is there a forced neighbor at (3,2,1)?
    # dx=1, dy=1. Check for obstacle at (x-dx, y) = (2,2,1). YES.
    # Check for clear path at (x-dx, y+dy) = (2,3,1). YES.
    # This constitutes a forced neighbor.
    assert jps._has_forced_neighbor(node=(3,2,1), direction=(1,1,0)) == True