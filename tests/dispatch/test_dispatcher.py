# FILE: tests/dispatch/test_dispatcher.py
import pytest
from unittest.mock import MagicMock

from dispatch.dispatcher import Dispatcher, get_hub_by_id
from system_state import get_initial_state
from config import HUBS
from ml_predictor.predictor import EnergyTimePredictor

@pytest.fixture
def mock_fleet_manager():
    return MagicMock()

@pytest.fixture
def mock_predictor():
    """Provides a mock EnergyTimePredictor that returns predictable values."""
    predictor = MagicMock(spec=EnergyTimePredictor)
    predictor.predict.return_value = (100.0, 20.0) # (time, energy)
    return predictor

@pytest.fixture
def dispatcher(mock_fleet_manager, mock_predictor):
    return Dispatcher(mock_fleet_manager, mock_predictor)

def test_smart_selection_chooses_best_drone(dispatcher, mock_fleet_manager, mock_predictor):
    """Assert that the drone with the highest score is chosen for a mission."""
    state = get_initial_state()
    hub_a_id = "HUB_A"
    
    state['drones']['Drone 1']['battery'] = 200.0
    state['drones']['Drone 1']['battery_health'] = 99.0
    for i in range(2, 6):
        state['drones'][f'Drone {i}']['battery_health'] = 90.0
    state['drones']['Drone 2']['battery'] = 180.0
    state['drones']['Drone 3']['battery'] = 10.0
    
    hub_a_loc = get_hub_by_id(hub_a_id)['location']
    order = {'id': 'O1', 'pos': (hub_a_loc[0] + 0.01, hub_a_loc[1], 50), 'payload_kg': 1.0}
    
    status = dispatcher.dispatch_order(state, order, hub_a_id)
    
    assert status == "dispatched"
    mock_fleet_manager.add_mission_to_queue.assert_called_once()
    call_args, _ = mock_fleet_manager.add_mission_to_queue.call_args
    mission_obj = call_args[0]
    assert mission_obj.drone_id == "Drone 1"

def test_out_of_range_order(dispatcher, mock_fleet_manager, mock_predictor):
    """Assert that an order is rejected if no drone has enough battery."""
    state = get_initial_state()
    hub_a_id = "HUB_A"
    
    mock_predictor.predict.return_value = (500.0, 150.0)
    
    for i in range(1, 6):
        state['drones'][f'Drone {i}']['battery'] = 200.0
        
    hub_a_loc = get_hub_by_id(hub_a_id)['location']
    order = {'id': 'O1', 'pos': (hub_a_loc[0] + 0.1, hub_a_loc[1], 50), 'payload_kg': 1.0}
    
    status = dispatcher.dispatch_order(state, order, hub_a_id)
    
    assert status == "out_of_range"
    mock_fleet_manager.add_mission_to_queue.assert_not_called()

def test_rebalancing_mission_creation(dispatcher, mock_fleet_manager):
    """Assert a rebalancing mission is correctly created and enqueued."""
    state = get_initial_state()
    hub_a_id = "HUB_A"
    hub_b_id = "HUB_B"
    
    state['drones']['Drone 5']['charge_cycles'] = 100
    state['drones']['Drone 4']['charge_cycles'] = 50
    
    dispatcher.create_rebalancing_mission(state, hub_a_id, hub_b_id)
    
    mock_fleet_manager.add_mission_to_queue.assert_called_once()
    call_args, _ = mock_fleet_manager.add_mission_to_queue.call_args
    mission_obj = call_args[0]
    
    assert mission_obj.mission_id.startswith("REBALANCE")
    assert mission_obj.drone_id == "Drone 5"
    assert mission_obj.start_hub == hub_a_id
    assert mission_obj.end_hub == hub_b_id
    assert not mission_obj.stops

def test_rejects_mission_with_insufficient_safety_margin(dispatcher, mock_fleet_manager, mock_predictor):
    """
    Tests that a drone is rejected if its battery is high enough for the raw trip,
    but not high enough to satisfy the safety margins.
    """
    state = get_initial_state()
    hub_a_id = "HUB_A"
    
    # Trip requires 40Wh (20 out, 20 back).
    # With RTH_FACTOR=1.5 and DISPATCH_MARGIN=1.2, total required is 40 * 1.5 * 1.2 = 72Wh.
    mock_predictor.predict.return_value = (100.0, 20.0)
    
    # This drone has enough for the raw trip (40Wh) but not for the safety margin (72Wh).
    state['drones']['Drone 1']['battery'] = 70.0 
    
    hub_a_loc = get_hub_by_id(hub_a_id)['location']
    order = {'id': 'O1', 'pos': (hub_a_loc[0] + 0.01, hub_a_loc[1], 50), 'payload_kg': 1.0}
    
    status = dispatcher.dispatch_order(state, order, hub_a_id)
    
    assert status == "out_of_range"
    mock_fleet_manager.add_mission_to_queue.assert_not_called()