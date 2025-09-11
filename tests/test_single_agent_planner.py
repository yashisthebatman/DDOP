# FILE: tests/test_single_agent_planner.py
import pytest
from unittest.mock import MagicMock, patch

from planners.single_agent_planner import SingleAgentPlanner
from environment import Environment
from utils.coordinate_manager import CoordinateManager

@pytest.fixture
def mock_env():
    env = MagicMock(spec=Environment)
    env.is_point_obstructed.return_value = False
    env.is_line_obstructed.return_value = False
    return env

@pytest.fixture
def planner(mock_env):
    coord_manager = CoordinateManager() # Use a real one
    predictor_mock = MagicMock()
    return SingleAgentPlanner(mock_env, predictor_mock, coord_manager)

@patch('planners.single_agent_planner.AnytimeRRTStar')
def test_replan_uses_local_rrt(mock_rrt, planner):
    """
    Tests that the tactical replan identifies a safe goal and calls a
    local RRT* to generate a detour.
    """
    # Mock the RRT* planner to return a simple detour
    mock_rrt.return_value.plan.return_value = ([(0,0,100), (450, 50, 100), (500,0,100)], "Success")

    drone_pos = (0, 0, 100)
    stale_path = [drone_pos, (250, 0, 100), (500, 0, 100), (750, 0, 100), (1000, 0, 100)]
    new_obstacle_bounds = (200, -50, 50, 300, 50, 150) # Blocks the (250,0,100) waypoint

    new_path, status = planner.perform_hybrid_replan(
        current_pos=drone_pos,
        stale_path=stale_path,
        new_obstacle_bounds=new_obstacle_bounds
    )

    assert status == "Hybrid replan successful."
    assert new_path is not None
    
    # Check that RRT* was called to plan from current pos to the first safe point (500,0,100)
    mock_rrt.assert_called_once()
    args, _ = mock_rrt.call_args
    assert args[0] == drone_pos         # start
    assert args[1] == (500, 0, 100)  # tactical_goal
    
    # Check that the path was spliced correctly
    # Expected: [(0,0,100), (450, 50, 100)] from detour + [(500,0,100), (750,0,100), (1000,0,100)] from original
    assert new_path[0] == (0,0,100)
    assert new_path[1] == (450, 50, 100)
    assert new_path[2] == (500, 0, 100)
    assert new_path[4] == (1000, 0, 100)
    assert len(new_path) == 5