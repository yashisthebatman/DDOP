# FILE: tests/test_operator_overrides.py

import pytest
from unittest.mock import MagicMock
from app import update_simulation
from dispatch.dispatcher import Dispatcher, MIN_ORDERS_FOR_BATCH
from system_state import get_initial_state

def test_pause_mission_halts_movement():
    """Test that pausing a mission prevents the drone from moving."""
    state = get_initial_state()
    state['simulation_time'] = 50.0
    
    drone_id = list(state['drones'].keys())[0]
    drone = state['drones'][drone_id]
    drone['status'] = 'EN ROUTE'
    drone['mission_id'] = 'M-123'
    drone['pos'] = (10, 10, 50)
    
    state['active_missions']['M-123'] = {
        'drone_id': drone_id, 'start_time': 0.0, 'total_planned_time': 100.0,
        'path_world_coords': [(10, 10, 50), (100, 100, 50)],
        'is_paused': True,  # Mission is paused
        'start_battery': 200,
        'total_planned_energy': 30
    }
    
    initial_pos = drone['pos']
    update_simulation(state, None) # Run one tick
    
    assert drone['pos'] == initial_pos # Position should not change

    # Now resume and check again
    state['active_missions']['M-123']['is_paused'] = False
    update_simulation(state, None)
    assert drone['pos'] != initial_pos # Position should now change

def test_high_priority_triggers_dispatcher():
    """Test that a high priority order bypasses the minimum batch size."""
    mock_vrp_solver = MagicMock()
    
    # FIX: The mock 'stops' list must contain full order dictionaries, including the 'pos' key.
    mock_stops = [{'id': 'O2', 'pos': (2,2,2), 'payload_kg': 1.0}]
    mock_vrp_solver.generate_tours.return_value = [{'drone_id': 'Drone 1', 'start_hub_id': 'Hub A (South Manhattan)', 'end_hub_id': 'Hub B (Midtown East)', 'stops': mock_stops, 'payload': 1.0}]
    
    dispatcher = Dispatcher(mock_vrp_solver)
    state = get_initial_state()
    
    # Ensure at least one drone is IDLE for the dispatcher to pick
    state['drones']['Drone 1']['status'] = 'IDLE'

    # Condition: 1 order, less than MIN_ORDERS_FOR_BATCH
    state['pending_orders'] = {'O1': {'id': 'O1', 'pos': (1,1,1), 'payload_kg': 1, 'high_priority': False}}
    
    dispatched = dispatcher.dispatch_missions(state)
    assert not dispatched
    mock_vrp_solver.generate_tours.assert_not_called()
    
    # Add a high priority order
    state['pending_orders']['O2'] = {'id': 'O2', 'pos': (2,2,2), 'payload_kg': 1, 'high_priority': True}
    
    dispatched = dispatcher.dispatch_missions(state)

    assert dispatched
    mock_vrp_solver.generate_tours.assert_called_once()