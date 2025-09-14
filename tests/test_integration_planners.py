# FILE: tests/test_integration_planners.py
import pytest
from unittest.mock import MagicMock
from planners.cbsh_planner import CBSHPlanner
from fleet.cbs_components import Agent
from environment import Environment
from utils.coordinate_manager import CoordinateManager
import numpy as np
from config import COARSE_GRID_RESOLUTION_M, GRID_VERTICAL_RESOLUTION_M, MIN_ALTITUDE

@pytest.fixture
def clear_env_and_coord_manager():
    """Provides a mock env with no obstacles and a real coordinate manager."""
    # FIX: Add all necessary methods to the spec
    env = MagicMock(spec=Environment, methods=['is_line_obstructed', 'is_point_obstructed', 'create_planning_grid', 'create_coarse_planning_grid'])
    env.is_line_obstructed.return_value = False
    env.is_point_obstructed.return_value = False
    
    coord_manager = CoordinateManager()
    
    fine_shape = (coord_manager.grid_width, coord_manager.grid_height, coord_manager.grid_depth)
    env.create_planning_grid.return_value = np.full(fine_shape, 1.0)
    
    w = int(coord_manager.area_width_m / COARSE_GRID_RESOLUTION_M)
    h = int(coord_manager.area_height_m / COARSE_GRID_RESOLUTION_M)
    d = int((coord_manager.alt_max - MIN_ALTITUDE) / GRID_VERTICAL_RESOLUTION_M)
    coarse_shape = (w, h, d)
    env.create_coarse_planning_grid.return_value = np.full(coarse_shape, True)
    
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
    assert np.allclose(path[0][0], agent1.start_pos, atol=1e-5)
    assert np.allclose(path[-1][0], agent1.goal_pos, atol=1e-5)