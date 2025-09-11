import pytest
import numpy as np
from unittest.mock import MagicMock

from planners.single_agent_planner import SingleAgentPlanner
from ml_predictor.predictor import EnergyTimePredictor
from environment import Environment
from utils.geometry import line_segment_intersects_aabb
from utils.coordinate_manager import CoordinateManager

@pytest.fixture
def mock_predictor():
    return MagicMock(spec=EnergyTimePredictor)

@pytest.fixture
def mock_coord_manager():
    return MagicMock(spec=CoordinateManager)

@pytest.fixture
def clear_environment():
    env = MagicMock(spec=Environment)
    env.is_point_obstructed.return_value = False
    env.is_line_obstructed.return_value = False
    return env

def test_replan_grid_scales_with_obstacle(mock_predictor, clear_environment):
    """
    Tests that the dynamic grid sizing logic is called with appropriate parameters.
    """
    coord_manager = CoordinateManager() # Use a real one to test its state changes
    # Spy on the method to see what it's called with
    coord_manager.set_local_grid_origin = MagicMock()

    planner = SingleAgentPlanner(clear_environment, mock_predictor, coord_manager)
    
    # Mock the D* Lite part to prevent full execution
    planner._find_tactical_detour = MagicMock(return_value=(None, "mocked"))

    drone_pos = (0, 0, 100)
    stale_path = [drone_pos, (1000, 0, 100)]
    # A large new obstacle
    new_obstacle_bounds = (400, -200, 50, 600, 200, 150)

    planner.perform_hybrid_replan(
        current_pos=drone_pos,
        stale_path=stale_path,
        new_obstacle_bounds=new_obstacle_bounds
    )
    
    # Check that D* Lite was called
    assert planner._find_tactical_detour.call_count == 1
    
    # Check that the grid was dynamically resized
    assert coord_manager.set_local_grid_origin.call_count == 1
    args, _ = coord_manager.set_local_grid_origin.call_args
    grid_center_world, grid_size_m = args
    
    # The grid size should be ~1.5x the path length (1000m)
    assert grid_size_m == pytest.approx(1500)
    # The center should be halfway along the path
    assert grid_center_world[0] == pytest.approx(500)