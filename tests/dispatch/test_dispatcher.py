# FILE: tests/dispatch/test_dispatcher.py
import pytest
from unittest.mock import MagicMock

from dispatch.dispatcher import Dispatcher, MIN_ORDERS_FOR_BATCH
from config import HUBS

@pytest.fixture
def mock_vrp_solver():
    return MagicMock()

@pytest.fixture
def dispatcher(mock_vrp_solver):
    return Dispatcher(mock_vrp_solver)

def get_base_state():
    """Returns a simple, default state for testing."""
    return {
        'drones': {
            # FIX: Use the full hub name from config to match application logic
            'D1': {'id': 'D1', 'status': 'IDLE', 'pos': (0,0,10), 'battery': 200.0, 'max_payload_kg': 5.0, 'home_hub': 'Hub A (South Manhattan)'},
            'D2': {'id': 'D2', 'status': 'RECHARGING', 'pos': (0,0,10), 'battery': 50.0, 'max_payload_kg': 5.0, 'home_hub': 'Hub A (South Manhattan)'}
        },
        'pending_orders': {
            'O1': {'id': 'O1', 'pos': (1,1,50), 'payload_kg': 1.0},
            'O2': {'id': 'O2', 'pos': (2,2,50), 'payload_kg': 1.0},
        },
        'active_missions': {}
    }

def test_dispatcher_does_not_trigger_below_threshold(dispatcher, mock_vrp_solver):
    """Test that the dispatcher doesn't run if there are too few orders."""
    state = get_base_state()
    # Ensure orders are below threshold for this specific test
    state['pending_orders'] = {k: v for i, (k, v) in enumerate(state['pending_orders'].items()) if i < MIN_ORDERS_FOR_BATCH - 1}

    dispatched = dispatcher.dispatch_missions(state)

    assert not dispatched
    mock_vrp_solver.generate_tours.assert_not_called()

def test_dispatcher_triggers_correctly(dispatcher, mock_vrp_solver):
    """Show that the dispatcher calls the solver only when trigger conditions are met."""
    state = get_base_state()
    # Add enough orders to trigger
    for i in range(MIN_ORDERS_FOR_BATCH):
        state['pending_orders'][f'O{i+3}'] = {'id': f'O{i+3}', 'pos': (i+3,i+3,50), 'payload_kg': 1.0}

    dispatcher.dispatch_missions(state)

    mock_vrp_solver.generate_tours.assert_called_once()
    # Check that only the IDLE drone was passed to the solver
    call_args, _ = mock_vrp_solver.generate_tours.call_args
    drones_arg = call_args[0]
    orders_arg = call_args[1]
    assert len(drones_arg) == 1
    assert drones_arg[0]['id'] == 'D1'
    # FIX: The total number of orders is 2 (from base state) + 5 (added) = 7.
    assert len(orders_arg) == 7

def test_state_updates_after_dispatch(dispatcher, mock_vrp_solver):
    """Assert that after a successful dispatch, orders are handled and the drone's status is updated to PLANNING."""
    state = get_base_state()
    for i in range(MIN_ORDERS_FOR_BATCH):
        state['pending_orders'][f'O{i+3}'] = {'id': f'O{i+3}', 'pos': (i+3,i+3,50), 'payload_kg': 1.0}
    
    # Mock the solver to return a single tour for D1
    mock_tour = [{
        'drone_id': 'D1',
        # FIX: Use the full, correct hub names to avoid the KeyError.
        'start_hub_id': 'Hub A (South Manhattan)',
        'end_hub_id': 'Hub B (Midtown East)',
        'stops': [state['pending_orders']['O1'], state['pending_orders']['O2']],
        'payload': 2.0
    }]
    mock_vrp_solver.generate_tours.return_value = mock_tour

    dispatched = dispatcher.dispatch_missions(state)

    assert dispatched
    # Check drone state
    assert state['drones']['D1']['status'] == 'PLANNING'
    assert state['drones']['D1']['mission_id'] is not None
    # Check that dispatched orders are still in pending (removed after successful planning)
    assert 'O1' in state['pending_orders']
    assert 'O2' in state['pending_orders']
    # Check mission state
    assert len(state['active_missions']) == 1
    mission_id = state['drones']['D1']['mission_id']
    mission = state['active_missions'][mission_id]
    assert mission['drone_id'] == 'D1'
    assert set(mission['order_ids']) == {'O1', 'O2'}
    assert mission['start_hub'] == 'Hub A (South Manhattan)'
    assert mission['end_hub'] == 'Hub B (Midtown East)'