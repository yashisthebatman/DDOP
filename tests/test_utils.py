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
    p1_in, p2_in = (1, 1, 1), (9, 9, 9)
    assert line_segment_intersects_aabb(p1_in, p2_in, box_bounds)
    p1_out, p2_out = (-1, 5, 5), (11, 5, 5)
    assert line_segment_intersects_aabb(p1_out, p2_out, box_bounds)
    p1_far, p2_far = (20, 20, 20), (30, 30, 30)
    assert not line_segment_intersects_aabb(p1_far, p2_far, box_bounds)
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

def test_world_to_meters_to_grid(coord_manager):
    """
    FIX: Rewritten test to validate the new, simpler CoordinateManager logic.
    It checks the conversion from world coordinates to the fixed grid.
    """
    # Point at the origin of the coordinate system
    origin_world = (coord_manager.lon_min, coord_manager.lat_min, coord_manager.alt_min)
    origin_meters = coord_manager.world_to_meters(origin_world)
    origin_grid = coord_manager.meters_to_grid(origin_meters)
    
    assert origin_meters[0] == pytest.approx(0)
    assert origin_meters[1] == pytest.approx(0)
    assert origin_grid == (0, 0, 0)

    # A point one grid cell away
    one_grid_away_m = (GRID_RESOLUTION_M, GRID_RESOLUTION_M, coord_manager.alt_min)
    one_grid_away_grid = coord_manager.meters_to_grid(one_grid_away_m)
    assert one_grid_away_grid == (1, 1, 0)
    
def test_reversibility_grid_to_meters_to_world(coord_manager):
    """
    FIX: Rewritten test for the new API. Checks that converting a grid point
    to meters and then to world coordinates is a consistent process.
    """
    grid_point = (5, 3, 10)
    
    # Convert to meters
    meters_point = coord_manager.grid_to_meters(grid_point)
    assert meters_point is not None
    
    # Convert back to grid and check
    reconverted_grid = coord_manager.meters_to_grid(meters_point)
    assert reconverted_grid == grid_point

    # Convert to world
    world_point = coord_manager.meters_to_world(meters_point)
    assert world_point is not None
    
    # Convert back to meters and check
    reconverted_meters = coord_manager.world_to_meters(world_point)
    assert reconverted_meters[0] == pytest.approx(meters_point[0])
    assert reconverted_meters[1] == pytest.approx(meters_point[1])