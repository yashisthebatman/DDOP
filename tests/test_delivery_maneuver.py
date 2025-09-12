# FILE: tests/test_delivery_maneuver.py

import pytest
import numpy as np
from unittest.mock import MagicMock

# --- System components needed for the test ---
from system_state import get_initial_state
from config import DELIVERY_MANEUVER_TIME_SEC, SIMULATION_TIME_STEP
from utils.geometry import calculate_distance_3d
from utils.coordinate_manager import CoordinateManager

# --- A simplified, local version of the app's update_simulation function ---
def simplified_update_simulation(state, coord_manager):
    """A focused version of update_simulation for testing delivery maneuvers."""
    state['simulation_time'] += SIMULATION_TIME_STEP

    for mission_id, mission in list(state['active_missions'].items()):
        drone_id = mission['drone_id']
        drone = state['drones'][drone_id]

        if drone['status'] == 'PERFORMING_DELIVERY':
            if state['simulation_time'] >= drone.get('maneuver_complete_at', float('inf')):
                mission['current_stop_index'] += 1
                drone['status'] = 'EN ROUTE'
                drone.pop('maneuver_complete_at', None)
            else:
                continue

        if drone['status'] == 'EN ROUTE':
            num_stops = len(mission.get('stops', []))
            stop_idx = mission.get('current_stop_index', 0)
            if stop_idx < num_stops:
                target_pos = mission['stops'][stop_idx]['pos']
                dist_m = calculate_distance_3d(
                    coord_manager.world_to_meters(drone['pos']),
                    coord_manager.world_to_meters(target_pos)
                )
                if dist_m < 5.0:
                    drone['status'] = 'PERFORMING_DELIVERY'
                    drone['maneuver_complete_at'] = state['simulation_time'] + DELIVERY_MANEUVER_TIME_SEC
                    drone['pos'] = target_pos
                    continue

@pytest.fixture
def test_state():
    """Provides a clean state with one active mission for testing."""
    state = get_initial_state()
    drone_id = "Drone 1"
    mission_id = "M-TEST"
    destination = (-74.0, 40.7, 100.0)

    state['drones'][drone_id]['status'] = 'EN ROUTE'
    state['drones'][drone_id]['mission_id'] = mission_id
    state['active_missions'][mission_id] = {
        'drone_id': drone_id,
        'stops': [{'id': 'Order1', 'pos': destination}],
        'current_stop_index': 0,
        'mission_time_elapsed': 0.0
    }
    return state

def test_drone_enters_delivery_state_on_arrival(test_state):
    """Manually place a drone at its destination and assert its status changes."""
    state = test_state
    drone_id = "Drone 1"
    destination = state['active_missions']['M-TEST']['stops'][0]['pos']
    coord_manager = CoordinateManager()
    
    # Place drone exactly at the destination
    state['drones'][drone_id]['pos'] = destination
    
    simplified_update_simulation(state, coord_manager)
    
    drone = state['drones'][drone_id]
    assert drone['status'] == 'PERFORMING_DELIVERY'
    assert 'maneuver_complete_at' in drone
    assert drone['maneuver_complete_at'] == pytest.approx(state['simulation_time'] + DELIVERY_MANEUVER_TIME_SEC)

def test_delivery_maneuver_has_correct_duration(test_state):
    """Test that the drone remains in the delivery state for the correct duration."""
    state = test_state
    drone_id = "Drone 1"
    coord_manager = CoordinateManager()
    
    # Manually put the drone into the delivery state
    state['drones'][drone_id]['status'] = 'PERFORMING_DELIVERY'
    state['drones'][drone_id]['maneuver_complete_at'] = state['simulation_time'] + DELIVERY_MANEUVER_TIME_SEC
    
    # Run simulation for less than the maneuver time
    for _ in range(5): # 5 * 0.5s = 2.5s
        simplified_update_simulation(state, coord_manager)
        
    drone = state['drones'][drone_id]
    assert drone['status'] == 'PERFORMING_DELIVERY'
    
    # Run simulation for more than the maneuver time
    state['simulation_time'] = DELIVERY_MANEUVER_TIME_SEC + 1
    simplified_update_simulation(state, coord_manager)
    
    drone = state['drones'][drone_id]
    mission = state['active_missions']['M-TEST']
    assert drone['status'] == 'EN ROUTE'
    assert 'maneuver_complete_at' not in drone
    assert mission['current_stop_index'] == 1