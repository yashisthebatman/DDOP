# FILE: tests/test_delivery_maneuver.py

import pytest
import numpy as np
from unittest.mock import MagicMock

from system_state import get_initial_state
from config import DELIVERY_MANEUVER_TIME_SEC, SIMULATION_TIME_STEP, HUBS
from utils.geometry import calculate_distance_3d
from utils.coordinate_manager import CoordinateManager
from server import update_simulation

@pytest.fixture
def test_state():
    """Provides a clean state with one active mission for testing."""
    state = get_initial_state()
    drone_id = "Drone 1"
    mission_id = "M-TEST"
    destination = (-74.0, 40.7, 100.0)
    end_hub_pos = HUBS["Hub A (South Manhattan)"]

    state['drones'][drone_id]['status'] = 'EN ROUTE'
    state['drones'][drone_id]['mission_id'] = mission_id
    state['drones'][drone_id]['pos'] = (-74.0, 40.7, 150.0) 
    state['active_missions'][mission_id] = {
        'mission_id': mission_id, 'drone_id': drone_id, 'order_ids': ['Order1'],
        'stops': [{'id': 'Order1', 'pos': destination}], 'current_stop_index': 0,
        'mission_time_elapsed': 0.0, 'flight_time_elapsed': 0.0, 'total_maneuver_time': DELIVERY_MANEUVER_TIME_SEC,
        'start_battery': 200, 'total_planned_energy': 20, 'total_planned_time': 1000,
        'destinations': [destination, end_hub_pos],
        'end_hub': "Hub A (South Manhattan)",
        'path_world_coords': [(-74.0, 40.7, 150.0), destination, end_hub_pos],
        'current_path_target_index': 1
    }
    return state

@pytest.fixture
def mock_planners():
    """Provides a mock planners dict needed by update_simulation."""
    coord_manager = CoordinateManager()
    return {"coord_manager": coord_manager}

def test_drone_enters_delivery_state_on_arrival(test_state, mock_planners):
    """Manually place a drone near its destination and assert its status changes."""
    state = test_state
    drone_id = "Drone 1"
    destination = state['active_missions']['M-TEST']['stops'][0]['pos']
    
    state['drones'][drone_id]['pos'] = (destination[0], destination[1], destination[2] + 2.0)
    
    update_simulation(state, mock_planners)
    
    drone = state['drones'][drone_id]
    mission = state['active_missions']['M-TEST']
    assert drone['status'] == 'PERFORMING_DELIVERY'
    assert 'maneuver_complete_at' in mission

def test_delivery_maneuver_has_correct_duration(test_state, mock_planners):
    """Test that the drone remains in the delivery state for the correct duration."""
    state = test_state
    drone_id = "Drone 1"
    
    state['drones'][drone_id]['status'] = 'PERFORMING_DELIVERY'
    # Use the mission dict for maneuver time, not the drone dict
    state['active_missions']['M-TEST']['maneuver_complete_at'] = state['simulation_time'] + DELIVERY_MANEUVER_TIME_SEC
    
    for _ in range(int((DELIVERY_MANEUVER_TIME_SEC - 5) / SIMULATION_TIME_STEP)): 
        update_simulation(state, mock_planners)
        
    drone = state['drones'][drone_id]
    assert drone['status'] == 'PERFORMING_DELIVERY'
    
    # Fast-forward time to just after the maneuver should complete
    state['simulation_time'] = state['active_missions']['M-TEST']['maneuver_complete_at'] + SIMULATION_TIME_STEP
    update_simulation(state, mock_planners)
    
    drone = state['drones'][drone_id]
    assert drone['status'] == 'RETURNING_TO_HUB'