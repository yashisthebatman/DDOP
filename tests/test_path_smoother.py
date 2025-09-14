import pytest
import numpy as np
from unittest.mock import MagicMock
from utils.path_smoother import PathSmoother

@pytest.fixture
def smoother():
    return PathSmoother()

def test_smoothing_returns_more_points(smoother):
    """Smoothed path should be denser than the original grid path."""
    path = [(0,0,10), (10,0,10), (10,10,10), (20, 10, 10)]
    env_mock = MagicMock()
    # FIX: The smoother now checks points, not just lines.
    env_mock.is_line_obstructed.return_value = False
    env_mock.is_point_obstructed.return_value = False
    smoothed_path = smoother.smooth_path(path, env_mock)
    assert len(smoothed_path) > len(path)
    
    assert np.allclose(smoothed_path[0], path[0])
    assert np.allclose(smoothed_path[-1], path[-1])

def test_smoother_avoids_static_collisions(smoother):
    """Smoother should not create a path that collides with an obstacle."""
    path = [(0,0,10), (10,0,10)]
    env_mock = MagicMock()
    # FIX: Use is_point_obstructed for the mock's side effect
    env_mock.is_point_obstructed.side_effect = lambda p: p[1] < -1 # Obstacle is at y < -1
    
    smoothed_path = smoother.smooth_path(path, env_mock)
    
    # Validate that no point in the new path triggered the collision
    for point in smoothed_path:
        assert not point[1] < -1

def test_smoother_introduces_no_dynamic_collisions(smoother):
    """Final validation should catch if smoothed paths now intersect."""
    path_a = [(0,0,10), (1,0,10), (2,0,10)]
    path_b = [(1,-1,10), (1,0,10), (1,1,10)]
    
    smoothed_a = [(0,0,10), (0.5,0,10), (1,0,10), (1.5,0,10), (2,0,10)]
    smoothed_b = [(1,-1,10), (1,-0.5,10), (1,0,10), (1,0.5,10), (1,1,10)]
    
    solution = {
        'drone_a': smoothed_a,
        'drone_b': smoothed_b
    }
    
    assert not smoother.validate_smoothed_solution(solution)

    non_colliding_b = [(10,-1,10), (10,0,10), (10,1,10)]
    solution_ok = {'drone_a': path_a, 'drone_b': non_colliding_b}
    assert smoother.validate_smoothed_solution(solution_ok)