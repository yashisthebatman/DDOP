import pytest
from environment import Environment, WeatherSystem, Building
from config import MIN_ALTITUDE, MAX_ALTITUDE, NO_FLY_ZONES

@pytest.fixture
def fresh_environment():
    """Provides a clean, initialized Environment for each test."""
    return Environment(WeatherSystem())

def test_initialization(fresh_environment):
    """Test that the environment initializes with static obstacles."""
    env = fresh_environment
    # Buildings are random but seeded, so the number should be consistent.
    assert len(env.buildings) == 20
    # Checks if static NFZs are indexed. Total obstacles = buildings + static NFZs.
    assert len(env.obstacles) == 20 + len(NO_FLY_ZONES)
    assert not env.dynamic_nfzs

def test_point_obstruction(fresh_environment):
    """Test point collision detection with buildings and NFZs."""
    env = fresh_environment
    # Point inside the first static NFZ
    obstructed_point = (-74.00, 40.72, 50.0)
    assert env.is_point_obstructed(obstructed_point)

    # Point below minimum altitude
    too_low_point = (-74.00, 40.71, MIN_ALTITUDE - 1)
    assert env.is_point_obstructed(too_low_point)

    # A known clear point
    clear_point = (-73.99, 40.71, 100.0)
    assert not env.is_point_obstructed(clear_point)

def test_line_obstruction(fresh_environment):
    """Test line-of-sight checks."""
    env = fresh_environment
    p1_clear = (-73.99, 40.71, 100.0)
    p2_clear = (-73.99, 40.73, 100.0)
    assert not env.is_line_obstructed(p1_clear, p2_clear)

    # Line passing directly through the first static NFZ
    p_start_outside = (-74.015, 40.72, 50.0)
    p_end_outside = (-73.990, 40.72, 50.0)
    assert env.is_line_obstructed(p_start_outside, p_end_outside)

def test_dynamic_nfz_management(fresh_environment):
    """Test the addition and removal of dynamic obstacles."""
    env = fresh_environment
    initial_obstacle_count = len(env.obstacles)
    
    # Simulate a dynamic event from the app
    env.update_environment(simulation_time=20, time_step=1)
    
    assert len(env.dynamic_nfzs) == 1
    assert len(env.obstacles) == initial_obstacle_count + 1
    assert env.was_nfz_just_added
    
    # Check that the new zone obstructs points
    point_in_dynamic_nfz = (-74.00, 40.725, 50.0)
    assert env.is_point_obstructed(point_in_dynamic_nfz)
    
    # Test removal
    env.remove_dynamic_obstacles()
    assert len(env.dynamic_nfzs) == 0
    assert len(env.obstacles) == initial_obstacle_count
    assert not env.is_point_obstructed(point_in_dynamic_nfz)