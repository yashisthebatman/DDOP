# FILE: tests/test_delivery_maneuver.py
import pytest
import numpy as np
from unittest.mock import MagicMock

from system_state import get_initial_state
from config import DELIVERY_MANEUVER_TIME_SEC, SIMULATION_TIME_STEP, HUBS
from utils.coordinate_manager import CoordinateManager
from server import update_simulation
from ml_predictor.predictor import EnergyTimePredictor # Import for spec

def get_hub_by_id(hub_id):
    return next((h for h in HUBS if h['id'] == hub_id), None)

@pytest.fixture
def test_state():
    state = get_initial_state()
    drone_id, mission_id = "Drone 1", "M-TEST"
    destination = (-74.0, 40.7, 100.0)
    end_hub_pos = get_hub_by_id("HUB_A")['location']
    state['drones'][drone_id].update({'status': 'EN ROUTE', 'mission_id': mission_id, 'pos': (-74.0, 40.7, 150.0)})
    state['active_missions'][mission_id] = {
        'mission_id': mission_id, 'drone_id': drone_id, 'order_ids': ['Order1'],
        'stops': [{'id': 'Order1', 'pos': destination}], 'current_stop_index': 0,
        'mission_time_elapsed': 0.0, 'flight_time_elapsed': 0.0, 'total_maneuver_time': DELIVERY_MANEUVER_TIME_SEC,
        'start_battery': 200, 'total_planned_energy': 20, 'total_planned_time': 1000,
        'destinations': [destination, end_hub_pos], 'end_hub': "HUB_A",
        'path_world_coords': [(-74.0, 40.7, 150.0), destination, end_hub_pos],
        'current_path_target_index': 1
    }
    return state

@pytest.fixture
def mock_planners():
    env_mock = MagicMock()
    env_mock.weather = MagicMock() # Add nested weather mock
    env_mock.weather.get_wind_at_location.return_value = np.array([0,0,0])
    env_mock.was_nfz_just_added = False
    
    # FIX: Create a fully structured mock to prevent ValueError
    predictor_mock = MagicMock(spec=EnergyTimePredictor)
    predictor_mock.fallback_predictor = MagicMock()
    predictor_mock.fallback_predictor.predict.return_value = (0.1, 0.1) 

    return {
        "coord_manager": CoordinateManager(),
        "env": env_mock,
        "predictor": predictor_mock
    }

def test_drone_enters_delivery_state_on_arrival(test_state, mock_planners):
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
    state = test_state
    drone_id = "Drone 1"
    state['drones'][drone_id]['status'] = 'PERFORMING_DELIVERY'
    state['active_missions']['M-TEST']['maneuver_complete_at'] = state['simulation_time'] + DELIVERY_MANEUVER_TIME_SEC
    for _ in range(int((DELIVERY_MANEUVER_TIME_SEC - 5) / SIMULATION_TIME_STEP)):
        update_simulation(state, mock_planners)
    assert state['drones'][drone_id]['status'] == 'PERFORMING_DELIVERY'
    state['simulation_time'] = state['active_missions']['M-TEST']['maneuver_complete_at'] + SIMULATION_TIME_STEP
    update_simulation(state, mock_planners)
    assert state['drones'][drone_id]['status'] == 'RETURNING_TO_HUB'