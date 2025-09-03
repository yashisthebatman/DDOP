# tests/test_path_planner.py

import pytest
import numpy as np
from unittest.mock import MagicMock

from path_planner import PathPlanner3D
from ml_predictor.predictor import EnergyTimePredictor
from environment import Environment, WeatherSystem
from utils.geometry import line_segment_intersects_aabb
from utils.coordinate_manager import CoordinateManager

@pytest.fixture
def mock_predictor():
    predictor = MagicMock(spec=EnergyTimePredictor)
    predictor.predict.return_value = (1.0, 1.0)
    return predictor

@pytest.fixture
def mock_coord_manager_for_planner():
    manager = MagicMock(spec=CoordinateManager)
    manager.world_to_local_meters.side_effect = lambda p: p
    def mock_grid_to_world(grid_pos=None, base_world_pos=None, offset_m=None):
        if base_world_pos is not None and offset_m is not None:
            return tuple(np.array(base_world_pos) + np.array(offset_m))
        return None
    manager.local_grid_to_world.side_effect = mock_grid_to_world
    return manager

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

def test_basic_pathfinding(mock_predictor, clear_environment, mock_coord_manager_for_planner, monkeypatch):
    monkeypatch.setattr('utils.rrt_star.RRT_ITERATIONS', 200)
    planner = PathPlanner3D(clear_environment, mock_predictor, mock_coord_manager_for_planner)
    start, end = (-0.5, 0, 100), (0.5, 0, 100)
    path, status = planner.find_path(start, end, 1.0, "time")
    
    assert path is not None
    assert status == "Path found successfully."
    assert path[-1] == end

def test_pathfinding_around_obstacle(mock_predictor, wall_environment, mock_coord_manager_for_planner, monkeypatch):
    # --- FIX: Increase iterations AND goal bias for reliability ---
    monkeypatch.setattr('utils.rrt_star.RRT_ITERATIONS', 1500)
    monkeypatch.setattr('utils.rrt_star.RRT_GOAL_BIAS', 0.2)
    planner = PathPlanner3D(wall_environment, mock_predictor, mock_coord_manager_for_planner)
    start, end = (-0.5, 0, 100), (0.5, 0, 100)
    path, status = planner.find_path(start, end, 1.0, "time")
    
    assert path is not None, f"Planner failed to find a path. Status: {status}"
    full_path = [start] + path
    for i in range(len(full_path) - 1):
        assert not wall_environment.is_line_obstructed(full_path[i], full_path[i+1])

# ... test_hybrid_replan_successful is unchanged ...
def test_hybrid_replan_successful(mock_predictor, clear_environment, mock_coord_manager_for_planner):
    """A critical unit test for the RRT*/D* Lite hybrid replan logic."""
    planner = PathPlanner3D(clear_environment, mock_predictor, mock_coord_manager_for_planner)
    
    drone_pos = (0, 0, 100)
    stale_path_from_drone = [
        drone_pos,
        (200, 0, 100),
        (400, 0, 100),
        (600, 0, 100)
    ]
    new_obstacle_bounds = (150, -50, 50, 250, 50, 150)

    mock_detour_path = [
        drone_pos,
        (100, 100, 100),
        (400, 0, 100)
    ]
    planner._find_tactical_detour = MagicMock(return_value=(mock_detour_path, "Path found."))

    new_path, status = planner.perform_hybrid_replan(
        current_pos=drone_pos,
        stale_path=stale_path_from_drone,
        new_obstacle_bounds=new_obstacle_bounds
    )

    assert new_path is not None
    assert status == "Hybrid replan successful."

    expected_path = [
        (0, 0, 100),
        (100, 100, 100),
        (400, 0, 100),
        (600, 0, 100)
    ]
    assert new_path == expected_path
    assert (200, 0, 100) not in new_path

def test_planner_integration_with_real_environment(mock_predictor, monkeypatch):
    monkeypatch.setattr('environment.NO_FLY_ZONES', [[-74.001, 40.70, -73.999, 40.74]])
    monkeypatch.setattr('environment.MAX_ALTITUDE', 200.0)
    monkeypatch.setattr('environment.Environment._generate_and_index_buildings', lambda self: [])
    
    # --- FIX: Increase iterations AND goal bias for reliability ---
    monkeypatch.setattr('utils.rrt_star.RRT_ITERATIONS', 2000)
    monkeypatch.setattr('utils.rrt_star.RRT_GOAL_BIAS', 0.25) # More aggressive bias
    monkeypatch.setattr('config.RRT_ITERATIONS', 2000)

    env = Environment(WeatherSystem())
    coord_manager = CoordinateManager()
    planner = PathPlanner3D(env, mock_predictor, coord_manager)

    start = (-74.01, 40.72, 50.0)
    end = (-73.99, 40.72, 50.0)

    path, status = planner.find_path(start, end, 1.0, "time")
    
    assert path is not None, f"Planner failed to find a path. Status: {status}"
    assert status == "Path found successfully."
    
    full_path = [start] + path
    for i in range(len(full_path) - 1):
        p1, p2 = full_path[i], full_path[i+1]
        assert not env.is_line_obstructed(p1, p2), f"Path segment {p1} -> {p2} is obstructed."

# ... test_planner_fails_if_start_or_goal_is_obstructed is unchanged ...
def test_planner_fails_if_start_or_goal_is_obstructed(mock_predictor):
    """
    Tests that the planner returns a clear failure message if the start or
    end point is inside an obstacle.
    """
    env = MagicMock(spec=Environment)
    coord_manager = CoordinateManager()
    planner = PathPlanner3D(env, mock_predictor, coord_manager)

    start, end = (0,0,50), (100,100,50)
    
    env.is_point_obstructed.side_effect = lambda p: p == start
    path, status = planner.find_path(start, end, 1.0, "time")
    assert path is None
    assert status == "Start point is obstructed."

    env.is_point_obstructed.side_effect = lambda p: p == end
    path, status = planner.find_path(start, end, 1.0, "time")
    assert path is None
    assert status == "Destination point is obstructed."