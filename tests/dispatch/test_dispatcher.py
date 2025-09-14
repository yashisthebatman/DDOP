# FILE: tests/dispatch/test_dispatcher.py
import pytest
from unittest.mock import MagicMock

from dispatch.dispatcher import Dispatcher, get_hub_by_id
from system_state import get_initial_state
from config import HUBS

@pytest.fixture
def mock_fleet_manager():
    return MagicMock()

@pytest.fixture
def dispatcher(mock_fleet_manager):
    return Dispatcher(mock_fleet_manager)

def test_smart_selection_chooses_best_drone(dispatcher, mock_fleet_manager):
    """Assert that the drone with the highest score is chosen for a mission."""
    state = get_initial_state()
    hub_a_id = "HUB_A"
    
    # Setup: Drone 1 is the best, Drone 2 is good, Drone 3 has low battery
    state['drones']['Drone 1']['battery'] = 200.0
    state['drones']['Drone 1']['battery_health'] = 99.0
    state['drones']['Drone 2']['battery'] = 180.0
    state['drones']['Drone 2']['battery_health'] = 95.0
    state['drones']['Drone 3']['battery'] = 10.0 # Not enough for the trip
    
    order = {'id': 'O1', 'pos': (0,0,50), 'payload_kg': 1.0}
    
    status = dispatcher.dispatch_order(state, order, hub_a_id)
    
    assert status == "dispatched"
    # The fleet manager's add_mission_to_queue method should have been called
    mock_fleet_manager.add_mission_to_queue.assert_called_once()
    # Check that the mission was assigned to the best drone
    call_args, _ = mock_fleet_manager.add_mission_to_queue.call_args
    mission_obj = call_args[0]
    assert mission_obj.drone_id == "Drone 1"

def test_out_of_range_order(dispatcher, mock_fleet_manager):
    """Assert that an order is rejected if no drone has enough battery."""
    state = get_initial_state()
    hub_a_id = "HUB_A"
    
    # Drain all batteries for drones at Hub A
    for i in range(1, 6):
        state['drones'][f'Drone {i}']['battery'] = 5.0
        
    order = {'id': 'O1', 'pos': (1000, 1000, 50), 'payload_kg': 1.0}
    
    status = dispatcher.dispatch_order(state, order, hub_a_id)
    
    assert status == "out_of_range"
    mock_fleet_manager.add_mission_to_queue.assert_not_called()

def test_rebalancing_mission_creation(dispatcher, mock_fleet_manager):
    """Assert a rebalancing mission is correctly created and enqueued."""
    state = get_initial_state()
    hub_a_id = "HUB_A"
    hub_b_id = "HUB_B"
    
    # Setup: Drone 5 is the "most used" at Hub A
    state['drones']['Drone 5']['charge_cycles'] = 100
    state['drones']['Drone 4']['charge_cycles'] = 50
    
    dispatcher.create_rebalancing_mission(state, hub_a_id, hub_b_id)
    
    mock_fleet_manager.add_mission_to_queue.assert_called_once()
    call_args, _ = mock_fleet_manager.add_mission_to_queue.call_args
    mission_obj = call_args[0]
    
    assert mission_obj.mission_id.startswith("REBALANCE")
    assert mission_obj.drone_id == "Drone 5" # Most used drone chosen
    assert mission_obj.start_hub == hub_a_id
    assert mission_obj.end_hub == hub_b_id
    assert not mission_obj.stops # No delivery stops