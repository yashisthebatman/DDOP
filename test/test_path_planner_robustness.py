# File: tests/test_path_planner_robustness.py
import pytest
from path_planner import PathPlanner3D
from environment import Environment, WeatherSystem
from ml_predictor.predictor import EnergyTimePredictor
from rtree import index # Import rtree to create the mock index

# FIX: Create a more robust Mock Environment that has all the attributes
# the PathPlanner3D constructor expects to find.
class MockEnv(Environment):
    def __init__(self):
        # We don't need a full weather system for these tests
        self.weather = None 
        
        # Provide the attributes that PathPlanner3D's __init__ will access
        self.static_nfzs = []
        self.dynamic_nfzs = []
        
        # Create a dummy obstacle_index (R-tree) so the attribute exists
        p = index.Property()
        p.dimension = 3
        self.obstacle_index = index.Index(properties=p)

    # This can be overridden in specific tests, but provides a default
    # behavior where nothing is obstructed. This allows the planner to initialize.
    def is_line_obstructed(self, p1, p2) -> bool:
        return False

    def is_point_obstructed(self, point):
        return False

@pytest.fixture
def planner():
    """Provides a PathPlanner3D instance with a controlled mock environment."""
    env = MockEnv()
    pred = EnergyTimePredictor()
    planner_instance = PathPlanner3D(env, pred)
    
    # Define a single obstructed grid point for our tests
    obstructed_grid_point = (50, 50, 5)
    
    # Override the planner's is_grid_obstructed method for fine-grained control
    # This is more direct than mocking the environment's methods for this specific test.
    def mock_is_grid_obstructed(pos):
        return pos == obstructed_grid_point
    
    planner_instance.is_grid_obstructed = mock_is_grid_obstructed
    
    return planner_instance, obstructed_grid_point

def test_find_nearest_cell_when_valid(planner):
    p, _ = planner
    valid_point = (10, 10, 1)
    # If the point is already valid, it should return itself
    assert p._find_nearest_valid_grid_cell(valid_point) == valid_point

def test_find_nearest_cell_when_obstructed(planner):
    p, obstructed_point = planner
    # If the point is obstructed, it should find a nearby valid point
    result = p._find_nearest_valid_grid_cell(obstructed_point)
    assert result is not None
    assert result != obstructed_point
    # A spiral search with radius 1 should find a direct neighbor
    assert abs(result[0] - obstructed_point[0]) <= 1
    assert abs(result[1] - obstructed_point[1]) <= 1
    assert abs(result[2] - obstructed_point[2]) <= 1

def test_find_path_fails_with_obstructed_start(planner):
    p, _ = planner
    
    # An arbitrary obstructed world point for the high-level check
    obstructed_world_point = (123.4, 567.8, 90.1)
    
    # Override the high-level environment check for this specific test
    p.env.is_point_obstructed = lambda point: point == obstructed_world_point

    # A far-away, valid destination
    end_pos = (234.5, 678.9, 100.0)

    path, status = p.find_path(
        start_pos=obstructed_world_point, 
        end_pos=end_pos,
        payload=1.0,
        mode='time'
    )
    
    assert path is None
    assert "Start point is located inside an obstacle" in status