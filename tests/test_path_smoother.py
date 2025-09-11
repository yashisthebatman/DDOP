import pytest
import numpy as np
from unittest.mock import MagicMock
from utils.path_smoother import PathSmoother

@pytest.fixture
def smoother():
    return PathSmoother()

def test_smoothing_returns_more_points(smoother):
    """Smoothed path should be denser than the original grid path."""
    path = [(0,0,10), (10,0,10), (10,10,10)]
    env_mock = MagicMock()
    # FIX: The test was failing because the mock environment was not configured.
    # A call to env_mock.is_line_obstructed() returns a new MagicMock, which is
    # truthy, causing the collision check to fail and the function to revert
    # to the original path. We must explicitly set the return value to False.
    env_mock.is_line_obstructed.return_value = False
    smoothed_path = smoother.smooth_path(path, env_mock)
    assert len(smoothed_path) > len(path)
    
    # Check start and end points are preserved
    assert np.allclose(smoothed_path[0], path[0])
    assert np.allclose(smoothed_path[-1], path[-1])

def test_smoother_avoids_static_collisions(smoother):
    """Smoother should not create a path that collides with an obstacle."""
    # Path that "cuts the corner" near an obstacle at (5, -5, 0)
    path = [(0,0,10), (10,0,10)]
    env_mock = MagicMock()
    # Mock a collision check that would fail if the path deviates too much
    env_mock.is_line_obstructed.side_effect = lambda p1, p2: p1[1] < 0 and p2[1] < 0
    
    smoothed_path = smoother.smooth_path(path, env_mock)
    
    # Validate that no segment of the new path triggered the collision
    for i in range(len(smoothed_path) - 1):
        p1, p2 = smoothed_path[i], smoothed_path[i+1]
        assert not (p1[1] < 0 and p2[1] < 0)

def test_smoother_introduces_no_dynamic_collisions(smoother):
    """Final validation should catch if smoothed paths now intersect."""
    # Two paths that pass close by at the same time
    # Path A: (0,0,10) -> (2,0,10) at t=0,1,2
    # Path B: (1,-1,10) -> (1,1,10) at t=0,1,2
    # They pass at (1,0,10) at t=1
    path_a = [(0,0,10), (1,0,10), (2,0,10)]
    path_b = [(1,-1,10), (1,0,10), (1,1,10)]
    
    # Let's say smoothing makes them both arrive at (1,0,10) earlier
    smoothed_a = [(0,0,10), (0.5,0,10), (1,0,10), (1.5,0,10), (2,0,10)] # Dense path, 5 points
    smoothed_b = [(1,-1,10), (1,-0.5,10), (1,0,10), (1,0.5,10), (1,1,10)] # Dense path, 5 points
    
    solution = {
        'drone_a': smoothed_a,
        'drone_b': smoothed_b
    }
    
    # Assuming each segment takes 1 time unit, they will collide at waypoint index 2
    assert not smoother.validate_smoothed_solution(solution)

    # A non-colliding example
    non_colliding_b = [(10,-1,10), (10,0,10), (10,1,10)]
    solution_ok = {'drone_a': path_a, 'drone_b': non_colliding_b}
    assert smoother.validate_smoothed_solution(solution_ok)