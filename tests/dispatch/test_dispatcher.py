# FILE: tests/dispatch/test_dispatcher.py
import pytest
from unittest.mock import MagicMock

from dispatch.dispatcher import Dispatcher
from config import HUBS, MIN_ORDERS_TO_DISPATCH

@pytest.fixture
def mock_vrp_solver():
    return MagicMock()

@pytest.fixture
def mock_fleet_manager():
    return MagicMock()

@pytest.fixture
def dispatcher(mock_vrp_solver, mock_fleet_manager):
    return Dispatcher(mock_vrp_solver, mock_fleet_manager)

def get_base_state():
    """Returns a simple, default state for testing."""
    return {
        'drones': {
            'D1': {'id': 'D1', 'status': 'IDLE', 'pos': (0,0,10), 'battery': 200.0, 'max_payload_kg': 5.0, 'home_hub': 'Hub A (South Manhattan)'},
            'D2': {'id': 'D2', 'status': 'RECHARGING', 'pos': (0,0,10), 'battery': 50.0, 'max_payload_kg': 5.0, 'home_hub': 'Hub A (South Manhattan)'}
        },
        'pending_orders': {
            'O1': {'id': 'O1', 'pos': (1,1,50), 'payload_kg': 1.0},
            'O2': {'id': 'O2', 'pos': (2,2,50), 'payload_kg': 1.0},
        },
        'active_missions': {}
    }

def test_dispatcher_does_not_trigger_below_threshold(dispatcher, mock_vrp_solver, monkeypatch):
    """Test that the dispatcher doesn't run if there are too few orders."""
    # Set threshold high to ensure it doesn't trigger
    monkeypatch.setattr('dispatch.dispatcher.MIN_ORDERS_TO_DISPATCH', 5)
    state = get_base_state()
    state['pending_orders'] = {'O1': {'id': 'O1', 'pos': (1,1,50), 'payload_kg': 1.0}}

    dispatched = dispatcher.dispatch_missions(state)

    assert not dispatched
    mock_vrp_solver.generate_tours.assert_not_called()

def test_dispatcher_triggers_on_single_order(dispatcher, mock_vrp_solver, mock_fleet_manager, monkeypatch):
    """Test that a single order can be dispatched if the config allows it."""
    monkeypatch.setattr('dispatch.dispatcher.MIN_ORDERS_TO_DISPATCH', 1)
    state = get_base_state()
    state['pending_orders'] = {'O1': {'id': 'O1', 'pos': (1,1,50), 'payload_kg': 1.0}}

    # Mock solver to return a valid tour for the single order
    mock_vrp_solver.generate_tours.return_value = [{
        'drone_id': 'D1', 'start_hub_id': 'Hub A (South Manhattan)', 'end_hub_id': 'Hub B (Midtown East)',
        'stops': [state['pending_orders']['O1']], 'payload': 1.0
    }]
    
    dispatched = dispatcher.dispatch_missions(state)

    assert dispatched
    mock_vrp_solver.generate_tours.assert_called_once()
    mock_fleet_manager.add_mission_to_queue.assert_called_once()

def test_dispatcher_triggers_correctly(dispatcher, mock_vrp_solver, monkeypatch):
    """Show that the dispatcher calls the solver when trigger conditions are met."""
    monkeypatch.setattr('dispatch.dispatcher.MIN_ORDERS_TO_DISPATCH', 2)
    state = get_base_state()
    
    dispatcher.dispatch_missions(state)

    mock_vrp_solver.generate_tours.assert_called_once()
    call_args, _ = mock_vrp_solver.generate_tours.call_args
    drones_arg = call_args[0]
    orders_arg = call_args[1]
    assert len(drones_arg) == 1
    assert drones_arg[0]['id'] == 'D1'
    assert len(orders_arg) == 2

def test_mission_is_enqueued_after_dispatch(dispatcher, mock_vrp_solver, mock_fleet_manager, monkeypatch):
    """Assert that after a successful dispatch, a mission is enqueued."""
    monkeypatch.setattr('dispatch.dispatcher.MIN_ORDERS_TO_DISPATCH', 1)
    state = get_base_state()
    
    mock_tour = [{
        'drone_id': 'D1',
        'start_hub_id': 'Hub A (South Manhattan)', 'end_hub_id': 'Hub B (Midtown East)',
        'stops': [state['pending_orders']['O1']], 'payload': 1.0
    }]
    mock_vrp_solver.generate_tours.return_value = mock_tour

    dispatched = dispatcher.dispatch_missions(state)

    assert dispatched
    # The dispatcher should NOT change the drone's state directly anymore
    assert state['drones']['D1']['status'] == 'IDLE'
    # It should call the fleet manager to enqueue the mission
    mock_fleet_manager.add_mission_to_queue.assert_called_once()
    
    # Check that the mission object passed to the queue is correct
    call_args, _ = mock_fleet_manager.add_mission_to_queue.call_args
    mission_obj = call_args[0]
    assert mission_obj.drone_id == 'D1'
    assert mission_obj.order_ids == ['O1']