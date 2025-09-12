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
    coord_manager = CoordinateManager()
    grid_shape = (coord_manager.grid_width, coord_manager.grid_height, coord_manager.grid_depth)
    mock_grid = np.full(grid_shape, True)
    env.create_planning_grid.return_value = mock_grid
    return env

@patch('planners.cbsh_planner.PathTimingSolver')
@patch('planners.cbsh_planner.AnytimeRRTStar')
@patch('planners.cbsh_planner.AStarPlanner')
def test_solves_head_on_conflict(mock_astar, mock_rrt, mock_timing_solver, mock_env):
    coord_manager = CoordinateManager()
    cbsh_planner = CBSHPlanner(mock_env, coord_manager)
    
    start1, goal1 = (-74.01, 40.71, 50), (-73.99, 40.71, 50)
    start2, goal2 = (-73.99, 40.71, 50), (-74.01, 40.71, 50)
    agent1 = Agent(id=1, start_pos=start1, goal_pos=goal1, config={})
    agent2 = Agent(id=2, start_pos=start2, goal_pos=goal2, config={})

    # FIX: Simplify the A* mock to return a 2-point path. This ensures the geometric
    # path passed to the timing solver mock is consistent with the 2-point timed
    # path returned by that mock, preventing logic errors in the CBSH search.
    mock_astar.return_value.find_path.return_value = [
        cbsh_planner.coord_manager.meters_to_grid(cbsh_planner.coord_manager.world_to_meters(start1)),
        cbsh_planner.coord_manager.meters_to_grid(cbsh_planner.coord_manager.world_to_meters(goal1)),
    ]
    
    def rrt_plan_side_effect(start_pos, goal_pos, env, coord_manager):
        mock_instance = MagicMock()
        mock_instance.plan.return_value = ([start_pos, goal_pos], "Success")
        return mock_instance
    mock_rrt.side_effect = rrt_plan_side_effect

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
    
    solution = cbsh_planner.plan_fleet([agent1, agent2])
    
    assert solution is not None
    assert 1 in solution and 2 in solution
    
    assert solution[1][-1][1] > solution[2][-1][1]