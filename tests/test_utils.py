import pytest
import numpy as np

from utils.geometry import calculate_distance_3d, calculate_vector_angle_3d, line_segment_intersects_aabb
from utils.coordinate_manager import CoordinateManager
from config import GRID_RESOLUTION_M

# --- Geometry Tests ---

def test_calculate_distance_3d():
    p1, p2 = (0, 0, 0), (3, 4, 0)
    assert calculate_distance_3d(p1, p2) == 5.0
    p3, p4 = (1, 2, 3), (1, 2, 3)
    assert calculate_distance_3d(p3, p4) == 0.0

def test_calculate_vector_angle_3d():
    v1 = (1, 0, 0)
    v2 = (0, 1, 0) # 90 degrees
    assert calculate_vector_angle_3d(v1, v2) == pytest.approx(np.pi / 2)
    
    v3 = (1, 0, 0)
    v4 = (-1, 0, 0) # 180 degrees
    assert calculate_vector_angle_3d(v3, v4) == pytest.approx(np.pi)

def test_line_segment_intersects_aabb():
    box_bounds = (0, 0, 0, 10, 10, 10)
    
    # Line completely inside
    p1_in, p2_in = (1, 1, 1), (9, 9, 9)
    assert line_segment_intersects_aabb(p1_in, p2_in, box_bounds)

    # Line crossing through
    p1_out, p2_out = (-1, 5, 5), (11, 5, 5)
    assert line_segment_intersects_aabb(p1_out, p2_out, box_bounds)
    
    # Line completely outside
    p1_far, p2_far = (20, 20, 20), (30, 30, 30)
    assert not line_segment_intersects_aabb(p1_far, p2_far, box_bounds)

    # Line endpoint is on the box face
    p1_touch, p2_touch = (10, 5, 5), (15, 5, 5)
    assert line_segment_intersects_aabb(p1_touch, p2_touch, box_bounds)

# --- CoordinateManager Tests ---

@pytest.fixture
def coord_manager():
    return CoordinateManager()

def test_coord_manager_initialization(coord_manager):
    assert coord_manager.grid_width > 0
    assert coord_manager.grid_height > 0
    assert coord_manager.grid_depth > 0

def test_world_to_local_grid_conversion(coord_manager):
    """Test the dynamic grid conversion logic."""
    # Set the origin to a known point
    origin_world = (-74.0, 40.72, 100.0)
    coord_manager.set_local_grid_origin(origin_world)
    
    # Test the origin itself - should map to (0, 0, grid_z)
    origin_grid = coord_manager.world_to_local_grid(origin_world)
    assert origin_grid is not None
    assert origin_grid[0] == 0
    assert origin_grid[1] == 0
    
    # Test a point offset by the grid resolution
    offset_world = coord_manager.local_grid_to_world((1, 1, origin_grid[2]))
    assert offset_world is not None
    
    # Re-convert and check if it's close to (1, 1)
    offset_grid = coord_manager.world_to_local_grid(offset_world)
    assert offset_grid is not None
    assert offset_grid[0] == 1
    assert offset_grid[1] == 1
    
    # Test a point far outside the grid
    far_world = (-73.0, 41.0, 100.0)
    far_grid = coord_manager.world_to_local_grid(far_world)
    assert far_grid is None
    
def test_grid_to_world_reversibility(coord_manager):
    """Test that converting a point to grid and back yields the original point."""
    origin_world = (-74.0, 40.72, 100.0)
    coord_manager.set_local_grid_origin(origin_world)

    grid_point = (5, -3, 10)
    world_point = coord_manager.local_grid_to_world(grid_point)
    assert world_point is not None
    
    reconverted_grid_point = coord_manager.world_to_local_grid(world_point)
    assert reconverted_grid_point is not None
    
    assert reconverted_grid_point[0] == pytest.approx(grid_point[0], abs=1)
    assert reconverted_grid_point[1] == pytest.approx(grid_point[1], abs=1)
    assert reconverted_grid_point[2] == grid_point[2]