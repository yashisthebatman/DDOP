# File: tests/test_coordinate_manager.py
import pytest
from utils.coordinate_manager import CoordinateManager
from config import AREA_BOUNDS, MIN_ALTITUDE, MAX_ALTITUDE

@pytest.fixture
def manager():
    """Provides a default CoordinateManager instance for tests."""
    return CoordinateManager(grid_depth=10)

def test_initialization(manager):
    assert manager.grid_width > 0
    assert manager.grid_height > 0
    assert manager.grid_depth == 10
    assert manager.grid_dims == (manager.grid_width, manager.grid_height, 10)

def test_is_valid_grid_position(manager):
    assert manager.is_valid_grid_position((0, 0, 0)) == True
    assert manager.is_valid_grid_position((manager.grid_width - 1, manager.grid_height - 1, 9)) == True
    assert manager.is_valid_grid_position((-1, 0, 0)) == False
    assert manager.is_valid_grid_position((0, -1, 0)) == False
    assert manager.is_valid_grid_position((0, 0, -1)) == False
    assert manager.is_valid_grid_position((manager.grid_width, 0, 0)) == False
    assert manager.is_valid_grid_position((0, manager.grid_height, 0)) == False
    assert manager.is_valid_grid_position((0, 0, 10)) == False

def test_world_to_grid_clamping(manager):
    # Test coordinates far outside the area bounds
    lon_outside = AREA_BOUNDS[0] - 1.0
    lat_outside = AREA_BOUNDS[1] - 1.0
    alt_outside_low = MIN_ALTITUDE - 100
    alt_outside_high = MAX_ALTITUDE + 100

    # Should clamp to the minimum corner (0, 0, 0)
    assert manager.world_to_grid((lon_outside, lat_outside, alt_outside_low)) == (0, 0, 0)
    
    # Should clamp to the maximum corner
    max_grid = (manager.grid_width - 1, manager.grid_height - 1, manager.grid_depth - 1)
    assert manager.world_to_grid((AREA_BOUNDS[2] + 1.0, AREA_BOUNDS[3] + 1.0, alt_outside_high)) == max_grid

def test_world_grid_roundtrip(manager):
    # Test a point in the middle of the area
    world_pos = (
        AREA_BOUNDS[0] + (AREA_BOUNDS[2] - AREA_BOUNDS[0]) / 2,
        AREA_BOUNDS[1] + (AREA_BOUNDS[3] - AREA_BOUNDS[1]) / 2,
        MIN_ALTITUDE + (MAX_ALTITUDE - MIN_ALTITUDE) / 2
    )
    
    grid_pos = manager.world_to_grid(world_pos)
    reverted_world_pos = manager.grid_to_world(grid_pos)

    # Check that the reverted world position is close to the original
    # It won't be exact due to grid discretization
    assert reverted_world_pos[0] == pytest.approx(world_pos[0], abs=0.0001)
    assert reverted_world_pos[1] == pytest.approx(world_pos[1], abs=0.0001)
    assert reverted_world_pos[2] == pytest.approx(world_pos[2], abs=20.0) # Altitude has larger steps