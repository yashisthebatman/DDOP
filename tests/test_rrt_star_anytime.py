# FILE: tests/test_rrt_star_anytime.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from utils.rrt_star_anytime import AnytimeRRTStar
from utils.coordinate_manager import CoordinateManager
from environment import Environment

@pytest.fixture
def mock_env():
    """Provides a mock environment."""
    env = MagicMock(spec=Environment)
    env.is_line_obstructed.return_value = False
    return env

@pytest.fixture
def mock_coord_manager():
    """A mock CoordinateManager configured for RRT* tests."""
    manager = MagicMock(spec=CoordinateManager)
    # Simulate a 1-to-1 mapping for meter-based calculations
    manager.world_to_local_meters.side_effect = lambda p: np.array(p)
    manager.lon_deg_to_m = 1.0
    manager.lat_deg_to_m = 1.0
    return manager

def calculate_path_length(path):
    """Calculates the total Euclidean distance of a path."""
    if not path or len(path) < 2:
        return 0
    # FIX: Replaced the deprecated np.sum(generator) with Python's built-in sum().
    # This is the recommended fix from the warning message and is more idiomatic here.
    return sum(np.linalg.norm(np.array(p2) - np.array(p1)) for p1, p2 in zip(path[:-1], path[1:]))

def test_finds_path_in_clear_environment(mock_env, mock_coord_manager):
    """Tests that a path can be found in a simple, unobstructed case."""
    start, goal = (0, 0, 100), (500, 500, 100)
    rrt = AnytimeRRTStar(start, goal, mock_env, mock_coord_manager)
    
    path, status = rrt.plan(time_budget_s=0.2)
    
    assert path is not None
    assert "successfully" in status
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)

def test_returns_none_when_trapped(mock_env, mock_coord_manager):
    """Tests that the planner returns None if no path can be found within the budget."""
    # Block all paths
    mock_env.is_line_obstructed.return_value = True
    
    start, goal = (0, 0, 100), (500, 0, 100)
    rrt = AnytimeRRTStar(start, goal, mock_env, mock_coord_manager)
    
    path, status = rrt.plan(time_budget_s=0.1) # Short budget for a failing test
    
    assert path is None
    assert "No strategic path" in status

def test_path_improves_with_more_time(mock_env, mock_coord_manager):
    """
    Validates the 'anytime' property: a longer time budget should yield
    a better (shorter or equal length) path.
    """
    start, goal = (0, 0, 100), (800, 800, 100)
    
    # --- Plan with a short time budget ---
    rrt_short = AnytimeRRTStar(start, goal, mock_env, mock_coord_manager)
    path_short, _ = rrt_short.plan(time_budget_s=0.1)
    assert path_short is not None
    length_short = calculate_path_length(path_short)

    # --- Plan with a longer time budget ---
    rrt_long = AnytimeRRTStar(start, goal, mock_env, mock_coord_manager)
    path_long, _ = rrt_long.plan(time_budget_s=0.3)
    assert path_long is not None
    length_long = calculate_path_length(path_long)

    # The longer search should find a path that is at least as good as, and likely better than, the short one.
    assert length_long <= length_short