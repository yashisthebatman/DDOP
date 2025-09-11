# FILE: tests/test_integration_planners.py
import pytest
from unittest.mock import MagicMock
from planners.cbsh_planner import CBSHPlanner
from fleet.cbs_components import Agent
from environment import Environment
from utils.coordinate_manager import CoordinateManager

@pytest.fixture
def clear_env():
    """Provides an environment with no obstacles for predictable pathfinding."""
    env = MagicMock(spec=Environment)
    env.is_line_obstructed.return_value = False
    return env

@pytest.fixture
def real_planner(clear_env):
    """Provides a real, fully-functional planner instance."""
    coord_manager = CoordinateManager()
    return CBSHPlanner(clear_env, coord_manager)

def test_planner_integration_solves_simple_conflict(real_planner):
    """
    An integration test to ensure CBSH, AnytimeRRTStar, and PathTimingSolver
    work together to solve a real (though simple) conflict.
    """
    # Two agents starting far apart, aiming for the same goal.
    # Agent 2 starts slightly further away, so it should be forced to wait for Agent 1.
    agent1 = Agent(id=1, start_pos=(-74.01, 40.71, 50), goal_pos=(-74.00, 40.72, 50), config={})
    agent2 = Agent(id=2, start_pos=(-74.011, 40.71, 50), goal_pos=(-74.00, 40.72, 50), config={})
    
    solution = real_planner.plan_fleet([agent1, agent2])

    assert solution is not None
    assert 1 in solution and solution[1] is not None
    assert 2 in solution and solution[2] is not None

    path1 = solution[1]
    path2 = solution[2]
    
    # The arrival time is the timestamp of the last point in the path.
    arrival_time_1 = path1[-1][1]
    arrival_time_2 = path2[-1][1]
    
    # Agent 2 should arrive at the goal at or after Agent 1.
    assert arrival_time_2 >= arrival_time_1