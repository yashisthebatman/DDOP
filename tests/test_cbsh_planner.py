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
    env.coord_manager = coord_manager
    grid_shape = (coord_manager.grid_width, coord_manager.grid_height, coord_manager.grid_depth)
    mock_grid = np.full(grid_shape, True)
    env.create_planning_grid.return_value = mock_grid
    return env

@patch('planners.cbsh_planner.PathTimingSolver')
@patch('planners.cbsh_planner.AnytimeRRTStar')
@patch('planners.cbsh_planner.AStarPlanner')
def test_solves_crossing_conflict(mock_astar, mock_rrt, mock_timing_solver, mock_env):
    coord_manager = mock_env.coord_manager
    cbsh_planner = CBSHPlanner(mock_env, coord_manager)
    
    start1, goal1 = (-74.01, 40.71, 50), (-73.99, 40.71, 50)
    start2, goal2 = (-74.00, 40.70, 50), (-74.00, 40.72, 50)
    
    agent1 = Agent(id=1, start_pos=start1, goal_pos=goal1, config={})
    agent2 = Agent(id=2, start_pos=start2, goal_pos=goal2, config={})

    def astar_side_effect(grid, start_grid, goal_grid):
        return [start_grid, goal_grid]
    mock_astar.return_value.find_path.side_effect = astar_side_effect
    
    # FIX: The side effect function cannot use 'self'. Instead, it can access
    # the mock instance it's attached to, which has the start/goal attributes.
    def rrt_plan_side_effect(time_budget_s, custom_bounds_m):
        # 'mock_rrt.return_value' is the mock instance of AnytimeRRTStar.
        # Its constructor (__init__) will be called with start_world and goal_world,
        # which we can capture and store on the mock itself if needed, or just access
        # the attributes that the real class would have.
        rrt_instance = mock_rrt.return_value
        return ([rrt_instance.start_pos_world, rrt_instance.goal_pos_world], "Success")

    # When AnytimeRRTStar is instantiated, store start/goal on the mock instance
    def rrt_init_side_effect(start_world, goal_world, env, coord_manager):
        mock_instance = MagicMock()
        mock_instance.start_pos_world = start_world
        mock_instance.goal_pos_world = goal_world
        mock_instance.plan.side_effect = rrt_plan_side_effect
        return mock_instance

    mock_rrt.side_effect = rrt_init_side_effect

    path1_default = [(start1, 0), (goal1, 10)]
    path1_wait = [(start1, 0), (start1, 2), (goal1, 12)]
    path2_default = [(start2, 0), (goal2, 10)]

    def timing_side_effect(geom_path, constraints):
        is_agent1 = np.allclose(geom_path[0], start1)
        if is_agent1:
            is_constrained = any(c.agent_id == 1 for c in constraints)
            return path1_wait if is_constrained else path1_default
        else:
            return path2_default

    mock_timing_solver.return_value.find_timing.side_effect = timing_side_effect
    
    solution = cbsh_planner.plan_fleet([agent1, agent2])
    
    assert solution is not None
    assert 1 in solution and 2 in solution
    
    assert solution[2][-1][1] == 10
    assert solution[1][-1][1] == 12