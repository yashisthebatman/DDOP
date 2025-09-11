# FILE: tests/test_integration_planners.py
import pytest
from unittest.mock import MagicMock
from planners.cbsh_planner import CBSHPlanner
from fleet.cbs_components import Agent
from environment import Environment
from utils.coordinate_manager import CoordinateManager
import numpy as np

@pytest.fixture
def clear_env_and_coord_manager():
    """Provides a mock env with no obstacles and a real coordinate manager."""
    env = MagicMock(spec=Environment)
    env.is_line_obstructed.return_value = False
    env.is_point_obstructed.return_value = False
    
    coord_manager = CoordinateManager()
    
    # Mock the grid creation to return a fully open grid
    grid_shape = (coord_manager.grid_width, coord_manager.grid_height, coord_manager.grid_depth)
    env.create_planning_grid.return_value = np.full(grid_shape, True)
    
    env.coord_manager = coord_manager
    return env, coord_manager

@pytest.fixture
def real_planner(clear_env_and_coord_manager):
    """Provides a real, fully-functional hybrid planner instance."""
    env, coord_manager = clear_env_and_coord_manager
    return CBSHPlanner(env, coord_manager)

def test_hybrid_planner_finds_path(real_planner):
    """
    An integration test to ensure the full Hybrid A*/RRT* stack can find a path.
    """
    agent1 = Agent(id=1, start_pos=(-74.01, 40.71, 50), goal_pos=(-74.00, 40.72, 50), config={})
    
    solution = real_planner.plan_fleet([agent1])

    assert solution is not None
    assert 1 in solution and solution[1] is not None
    
    path = solution[1]
    # Check that path starts and ends correctly (approximate due to float precision)
    assert np.allclose(path[0][0], agent1.start_pos, atol=1e-5)
    assert np.allclose(path[-1][0], agent1.goal_pos, atol=1e-5)