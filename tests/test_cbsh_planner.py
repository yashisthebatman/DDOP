# FILE: tests/test_cbsh_planner.py
import pytest
from unittest.mock import MagicMock, patch
from fleet.cbs_components import Agent
from planners.cbsh_planner import CBSHPlanner
from utils.coordinate_manager import CoordinateManager
from environment import Environment
import numpy as np

@pytest.fixture
def mock_env():
    env = MagicMock(spec=Environment)
    env.is_point_obstructed.return_value = False
    return env

@pytest.fixture
def cbsh_planner(mock_env):
    coord_manager = CoordinateManager()
    planner = CBSHPlanner(mock_env, coord_manager)
    return planner

@patch('planners.cbsh_planner.PathTimingSolver')
@patch('planners.cbsh_planner.AnytimeRRTStar')
@patch('planners.cbsh_planner.AStarPlanner')
def test_solves_head_on_conflict(mock_astar, mock_rrt, mock_timing_solver, cbsh_planner):
    # Use realistic, in-bounds coordinates for the test agents
    start1, goal1 = (-74.01, 40.71, 50), (-73.99, 40.71, 50)
    start2, goal2 = (-73.99, 40.71, 50), (-74.01, 40.71, 50)
    agent1 = Agent(id=1, start_pos=start1, goal_pos=goal1, config={})
    agent2 = Agent(id=2, start_pos=start2, goal_pos=goal2, config={})

    # --- Mock the sub-planners to return predictable results ---
    # A* finds a simple path
    mock_astar.return_value.find_path.return_value = [(0,0,0), (1,1,1)]
    # RRT* finds a direct connection
    mock_rrt.return_value.plan.return_value = ([start1, goal1], "Success")
    
    # Mock timing solver with conflict resolution logic
    path1_default = [(start1, 0), (goal1, 10)]
    path1_wait = [(start1, 0), (start1, 2), (goal1, 12)]
    path2_default = [(start2, 0), (goal2, 10)]

    def timing_side_effect(geom_path, constraints):
        is_agent1 = np.allclose(geom_path[0], start1)
        is_constrained = any(c.agent_id == 1 for c in constraints)
        if is_agent1:
            return path1_wait if is_constrained else path1_default
        return path2_default
    
    mock_timing_solver.return_value.find_timing.side_effect = timing_side_effect
    
    # Run Planner
    solution = cbsh_planner.plan_fleet([agent1, agent2])
    
    assert solution is not None
    assert 1 in solution and 2 in solution
    
    # The planner should have constrained agent 1, making it wait.
    assert solution[1][-1][1] > solution[2][-1][1] # Agent 1 arrives later