import pytest
import numpy as np
from unittest.mock import MagicMock

from utils.rrt_star import RRTStar
from utils.coordinate_manager import CoordinateManager
from environment import Environment

@pytest.fixture
def mock_config(monkeypatch):
    # This fixture already keeps iterations reasonably low for most tests
    monkeypatch.setattr('utils.rrt_star.RRT_ITERATIONS', 500)
    monkeypatch.setattr('utils.rrt_star.RRT_STEP_SIZE_METERS', 150.0)
    monkeypatch.setattr('utils.rrt_star.RRT_GOAL_BIAS', 0.1)
    monkeypatch.setattr('utils.rrt_star.RRT_NEIGHBORHOOD_RADIUS_METERS', 200.0)
    monkeypatch.setattr('utils.rrt_star.AREA_BOUNDS', [-1, -1, 1, 1])
    monkeypatch.setattr('utils.rrt_star.MIN_ALTITUDE', 0)
    monkeypatch.setattr('utils.rrt_star.MAX_ALTITUDE', 200)

@pytest.fixture
def mock_coord_manager():
    """A mock CoordinateManager configured for RRT* tests."""
    manager = MagicMock(spec=CoordinateManager)
    manager.world_to_local_meters.side_effect = lambda p: p
    
    manager.lon_min = -1.0
    manager.lat_min = -1.0
    manager.lon_max = 1.0
    manager.lat_max = 1.0
    manager.lon_deg_to_m = 1.0 
    manager.lat_deg_to_m = 1.0

    def mock_grid_to_world(grid_pos=None, base_world_pos=None, offset_m=None):
        if base_world_pos is not None and offset_m is not None:
            return tuple(np.array(base_world_pos) + np.array(offset_m))
        return None
    manager.local_grid_to_world.side_effect = mock_grid_to_world
    return manager

def calculate_path_length(path):
    length = 0
    for i in range(len(path) - 1):
        length += np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
    return length

def test_steering(mock_config, mock_coord_manager):
    """Ensure the _steer function correctly extends a node by the exact step size."""
    env = MagicMock(spec=Environment)
    rrt = RRTStar(start=(0,0,0), goal=(1000,0,0), env=env, coord_manager=mock_coord_manager)
    
    new_pos = rrt._steer(from_pos=(0,0,0), to_sample=(1000,0,0))
    assert new_pos is not None
    distance = np.linalg.norm(np.array(new_pos))
    assert distance == pytest.approx(150.0)

    new_pos_short = rrt._steer(from_pos=(0,0,0), to_sample=(50,0,0))
    assert new_pos_short is not None
    distance_short = np.linalg.norm(np.array(new_pos_short))
    assert distance_short == pytest.approx(50.0)

def test_no_path_trapped(mock_config, mock_coord_manager):
    """RRT* should run for max iterations and gracefully return None if trapped."""
    env = MagicMock(spec=Environment)
    env.is_line_obstructed.return_value = True
    
    rrt = RRTStar(start=(0,0,100), goal=(500,0,100), env=env, coord_manager=mock_coord_manager)
    path, status = rrt.plan()

    assert path is None
    assert status == "No strategic path found."
    assert len(rrt.nodes) == 1

def test_path_optimality(mock_config, mock_coord_manager, monkeypatch):
    """A path found with more iterations should be shorter or equal in length."""
    env = MagicMock(spec=Environment)
    env.is_line_obstructed.return_value = False
    
    monkeypatch.setattr('utils.rrt_star.RRT_ITERATIONS', 200)
    rrt_low = RRTStar(start=(0,0,100), goal=(800, 800, 100), env=env, coord_manager=mock_coord_manager)
    path_low, _ = rrt_low.plan()
    assert path_low is not None
    length_low = calculate_path_length(path_low)

    # OPTIMIZATION: Reduce high iteration count to speed up test
    monkeypatch.setattr('utils.rrt_star.RRT_ITERATIONS', 800) # Reduced from 2000
    rrt_high = RRTStar(start=(0,0,100), goal=(800, 800, 100), env=env, coord_manager=mock_coord_manager)
    path_high, _ = rrt_high.plan()
    assert path_high is not None
    length_high = calculate_path_length(path_high)

    assert length_high <= length_low

def test_goal_bias(mock_config, mock_coord_manager, monkeypatch):
    """Test that a high goal bias finds a direct path quickly."""
    env = MagicMock(spec=Environment)
    env.is_line_obstructed.return_value = False
    
    monkeypatch.setattr('utils.rrt_star.RRT_GOAL_BIAS', 1.0)
    monkeypatch.setattr('utils.rrt_star.RRT_ITERATIONS', 50)

    rrt = RRTStar(start=(0,0,100), goal=(800,0,100), env=env, coord_manager=mock_coord_manager)
    path, status = rrt.plan()

    assert path is not None
    assert status == "Path found successfully."
    assert len(path) < 10