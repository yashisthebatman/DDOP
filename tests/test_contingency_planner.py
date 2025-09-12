# FILE: tests/test_contingency_planner.py
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

# Import the module to test
from simulation.contingency_planner import check_for_contingencies
from system_state import get_initial_state
from config import HUBS

@pytest.fixture
def mock_planners():
    """Provides a mocked planners dictionary for testing contingencies."""
    env = MagicMock()
    env.was_nfz_just_added = False
    env.is_line_obstructed.return_value = False
    
    predictor = MagicMock()
    # Default return: (time, energy)
    predictor.predict.return_value = (50.0, 20.0)

    coord_manager = MagicMock()
    # Simple 1:1000 mapping for predictable distance calculation
    coord_manager.world_to_meters.side_effect = lambda p: (p[0] * 1000, p[1] * 1000, p[2])
    
    # Mock the SingleAgentPlanner that will be instantiated inside the function
    mock_single_planner = MagicMock()
    mock_single_planner.find_strategic_path_rrt.return_value = ([(-74.0, 40.7, 50), (-74.0, 40.71, 50)], "Success")

    # Use patch as a context manager to intercept the class instantiation
    with patch('simulation.contingency_planner.SingleAgentPlanner', return_value=mock_single_planner) as mock_planner_class:
        yield {
            "env": env,
            "predictor": predictor,
            "coord_manager": coord_manager,
            "mock_planner_class": mock_planner_class,
            "mock_single_planner": mock_single_planner
        }

@pytest.fixture
def active_mission_state():
    """Provides a state with one active drone and mission."""
    state = get_initial_state()
    state['drones']['Drone 1']['status'] = 'EN ROUTE'
    state['drones']['Drone 1']['mission_id'] = 'M-123'
    state['drones']['Drone 1']['pos'] = (-74.0, 40.72, 50)
    
    state['active_missions']['M-123'] = {
        'mission_id': 'M-123',
        'drone_id': 'Drone 1',
        'order_ids': ['OrderX'],
        'stops': [{'id': 'OrderX', 'pos': (-74.0, 40.73, 50), 'payload_kg': 1.0}],
        'destinations': [(-74.0, 40.73, 50)],
        'path_world_coords': [(-74.0, 40.72, 50), (-74.0, 40.73, 50)],
        'start_time': 0,
        'payload_kg': 1.0,
        'start_battery': 200,
    }
    state['pending_orders'] = {}
    return state

def test_low_battery_triggers_return_to_hub(active_mission_state, mock_planners):
    state = active_mission_state
    # Set drone battery to a value that will trigger the contingency
    state['drones']['Drone 1']['battery'] = 30.0 # Critically low
    
    # Mock predictor to return high energy costs, ensuring a trigger.
    # energy_to_dest=20, energy_to_hub=20. Total required with buffer = (20+20)*1.1 = 44. Battery is 30.
    mock_planners['predictor'].predict.return_value = (50.0, 20.0)
    
    check_for_contingencies(state, mock_planners)
    
    drone = state['drones']['Drone 1']
    assert drone['status'] == 'EMERGENCY_RETURN'
    assert 'M-123' not in state['active_missions']
    assert 'OrderX' in state['pending_orders']
    assert len(state['completed_missions_log']) == 1
    assert state['completed_missions_log'][0]['outcome'] == 'Failed: Critically Low Battery'
    
    # Check that a new emergency mission was created
    assert drone['mission_id'].startswith('EM-')
    assert drone['mission_id'] in state['active_missions']
    mock_planners['mock_single_planner'].find_strategic_path_rrt.assert_called_once()

def test_new_nfz_triggers_replanning(active_mission_state, mock_planners):
    state = active_mission_state
    env = mock_planners['env']
    
    # Simulate a new NFZ being added
    env.was_nfz_just_added = True
    # Make the environment report an obstruction on the drone's path
    env.is_line_obstructed.return_value = True
    
    check_for_contingencies(state, mock_planners)
    
    drone = state['drones']['Drone 1']
    assert drone['status'] == 'EMERGENCY_RETURN'
    assert 'M-123' not in state['active_missions']
    assert 'OrderX' in state['pending_orders']
    assert len(state['completed_missions_log']) == 1
    assert state['completed_missions_log'][0]['outcome'] == 'Failed: Path Invalidated by NFZ'
    # The flag should be reset after the check is complete
    assert env.was_nfz_just_added is False