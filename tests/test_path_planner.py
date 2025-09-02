import pytest
import numpy as np
from unittest.mock import MagicMock

from path_planner import PathPlanner3D
from ml_predictor.predictor import EnergyTimePredictor
from environment import Environment
from utils.geometry import line_segment_intersects_aabb

@pytest.fixture
def mock_predictor():
    predictor = MagicMock(spec=EnergyTimePredictor)
    predictor.predict.return_value = (1.0, 1.0)
    return predictor

@pytest.fixture
def clear_environment():
    env = MagicMock(spec=Environment)
    env.is_point_obstructed.return_value = False
    env.is_line_obstructed.return_value = False
    return env

@pytest.fixture
def wall_environment():
    env = MagicMock(spec=Environment)
    env.is_point_obstructed.return_value = False
    wall_bounds = (-0.05, -0.8, 0, 0.05, 0.8, 180)
    env.is_line_obstructed.side_effect = lambda p1, p2: line_segment_intersects_aabb(p1, p2, wall_bounds)
    return env

def test_basic_pathfinding(mock_predictor, clear_environment):
    """Test basic RRT* pathfinding in an open environment."""
    planner = PathPlanner3D(clear_environment, mock_predictor)
    start, end = (-0.5, 0, 100), (0.5, 0, 100)
    path, status = planner.find_path(start, end, 1.0, "time")
    
    assert path is not None
    assert status == "Path found successfully."
    assert path[-1] == end

def test_pathfinding_around_obstacle(mock_predictor, wall_environment):
    """Test RRT* finding a path around a central wall."""
    planner = PathPlanner3D(wall_environment, mock_predictor)
    start, end = (-0.5, 0, 100), (0.5, 0, 100)
    path, status = planner.find_path(start, end, 1.0, "time")
    
    assert path is not None
    full_path = [start] + path
    for i in range(len(full_path) - 1):
        assert not wall_environment.is_line_obstructed(full_path[i], full_path[i+1])

def test_hybrid_replan_successful(mock_predictor, clear_environment):
    """A critical unit test for the RRT*/D* Lite hybrid replan logic."""
    planner = PathPlanner3D(clear_environment, mock_predictor)
    
    drone_pos = (0, 0, 100)
    stale_path_from_drone = [
        drone_pos,
        (200, 0, 100),  # This waypoint is now inside the new obstacle
        (400, 0, 100),  # This is the safe tactical goal
        (600, 0, 100)
    ]
    new_obstacle_bounds = (150, -50, 50, 250, 50, 150)

    # FIX: Mock the tactical planner to return a predictable detour.
    # This tests the stitching logic of perform_hybrid_replan in isolation.
    mock_detour_path = [
        drone_pos,          # Starts at drone
        (100, 100, 100),    # Goes "around"
        (400, 0, 100)       # Ends at the safe tactical goal
    ]
    planner._find_tactical_detour = MagicMock(return_value=(mock_detour_path, "Path found."))
    
    new_path, status = planner.perform_hybrid_replan(
        current_pos=drone_pos,
        stale_path=stale_path_from_drone,
        new_obstacle_bounds=new_obstacle_bounds
    )
    
    assert new_path is not None
    assert status == "Hybrid replan successful."
    
    # Expected path is the mock detour (minus its last element) + the rest of the stale path
    expected_path = [
        (0, 0, 100),
        (100, 100, 100),
        (400, 0, 100),
        (600, 0, 100)
    ]
    assert new_path == expected_path
    assert (200, 0, 100) not in new_path