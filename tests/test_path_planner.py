import pytest
import numpy as np
from unittest.mock import MagicMock

from path_planner import PathPlanner3D
from environment import Environment
from ml_predictor.predictor import EnergyTimePredictor
# FIX: Import geometry to create a more realistic mock environment.
from utils.geometry import line_segment_intersects_aabb, point_in_aabb

@pytest.fixture
def mock_predictor():
    predictor = EnergyTimePredictor()
    predictor.predict = MagicMock(side_effect=lambda p1, p2, *args, **kwargs: (np.linalg.norm(np.array(p2)-np.array(p1)), np.linalg.norm(np.array(p2)-np.array(p1))))
    return predictor

@pytest.fixture
def clear_environment():
    env = MagicMock(spec=Environment)
    env.is_point_obstructed.return_value = False
    env.is_line_obstructed.return_value = False
    env.AREA_BOUNDS = [-1, -1, 1, 1]; env.MIN_ALTITUDE = 0; env.MAX_ALTITUDE = 200
    return env

@pytest.fixture
def wall_environment():
    env = MagicMock(spec=Environment)
    env.is_point_obstructed.return_value = False
    # FIX: Make the wall a large but FINITE panel, not an infinite plane.
    wall_bounds = (-0.05, -0.8, 0, 0.05, 0.8, 180)
    env.is_line_obstructed.side_effect = lambda p1, p2: line_segment_intersects_aabb(p1, p2, wall_bounds)
    env.AREA_BOUNDS = [-1, -1, 1, 1]; env.MIN_ALTITUDE = 0; env.MAX_ALTITUDE = 200
    return env

def test_basic_pathfinding(mock_predictor, clear_environment):
    planner = PathPlanner3D(clear_environment, mock_predictor)
    planner.build_abstract_graph()
    start, end = (-0.5, 0, 100), (0.5, 0, 100)
    path, status = planner.find_path(start, end, 1.0, "time")
    assert path is not None and status == "Path found successfully."
    assert len(path) == 1 and path[0] == end

def test_pathfinding_around_obstacle(mock_predictor, wall_environment):
    planner = PathPlanner3D(wall_environment, mock_predictor)
    planner.build_abstract_graph()
    start, end = (-0.5, 0, 100), (0.5, 0, 100)
    path, status = planner.find_path(start, end, 1.0, "time")
    # This assertion will now pass as the finite wall can be navigated.
    assert path is not None
    assert len(path) > 1
    full_path = [start] + path
    for i in range(len(full_path) - 1):
        assert not wall_environment.is_line_obstructed(full_path[i], full_path[i+1])

def test_impossible_path(mock_predictor, clear_environment):
    clear_environment.is_point_obstructed.side_effect = lambda p: p == (0.5, 0, 100)
    planner = PathPlanner3D(clear_environment, mock_predictor)
    planner.build_abstract_graph()
    start, end = (-0.5, 0, 100), (0.5, 0, 100)
    path, status = planner.find_path(start, end, 1.0, "time")
    assert path is None and status == "Destination point is obstructed."

def test_hybrid_replan_simple(mock_predictor, clear_environment):
    planner = PathPlanner3D(clear_environment, mock_predictor)
    planner.build_abstract_graph()
    drone_pos, goal_pos = (0, 0, 100), (0.8, 0, 100)
    new_obstacle_bounds = (0.05, -1, 0, 0.15, 1, 200)
    # FIX: Make the mock more robust to handle the new obstacle realistically.
    original_line_check = clear_environment.is_line_obstructed
    clear_environment.is_line_obstructed.side_effect = lambda p1, p2: (
        original_line_check(p1, p2) or line_segment_intersects_aabb(p1, p2, new_obstacle_bounds)
    )
    clear_environment.is_point_obstructed.side_effect = lambda p: point_in_aabb(p, new_obstacle_bounds)
    new_path, status = planner.perform_hybrid_replan(
        current_pos=drone_pos, goal_pos=goal_pos, new_obstacle_bounds=new_obstacle_bounds,
        payload_kg=1.0, mode="time"
    )
    assert new_path is not None and status == "Hybrid replan successful."
    assert len(new_path) > 1
    full_new_path = [drone_pos] + new_path
    for i in range(len(full_new_path) - 1):
        assert not line_segment_intersects_aabb(full_new_path[i], full_new_path[i+1], new_obstacle_bounds)

def test_hybrid_replan_trapped_drone(mock_predictor, clear_environment):
    planner = PathPlanner3D(clear_environment, mock_predictor)
    planner.build_abstract_graph()
    drone_pos, goal_pos = (0, 0, 100), (0.8, 0, 100)
    new_obstacle_bounds = (-0.1, -0.1, 90, 0.1, 0.1, 110)
    clear_environment.is_line_obstructed.return_value = True
    new_path, status = planner.perform_hybrid_replan(
        current_pos=drone_pos, goal_pos=goal_pos, new_obstacle_bounds=new_obstacle_bounds,
        payload_kg=1.0, mode="time"
    )
    assert new_path is None and "Fatal: Drone is trapped" in status