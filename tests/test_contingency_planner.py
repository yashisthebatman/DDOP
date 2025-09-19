# FILE: tests/test_contingency_planner.py
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from simulation.contingency_planner import check_for_contingencies
from system_state import get_initial_state
from config import HUBS

@pytest.fixture
def mock_planners():
    env = MagicMock()
    env.weather = MagicMock()
    env.weather.get_wind_at_location.return_value = np.array([0,0,0])
    env.was_nfz_just_added = False
    env.is_line_obstructed.return_value = False
    predictor = MagicMock()
    predictor.predict.return_value = (50.0, 20.0)
    coord_manager = MagicMock()
    coord_manager.world_to_meters.side_effect = lambda p: (p[0] * 1000, p[1] * 1000, p[2])
    mock_single_planner = MagicMock()
    mock_single_planner.find_strategic_path_rrt.return_value = ([(-74.0, 40.7, 50), (-74.0, 40.71, 50)], "Success")
    with patch('simulation.contingency_planner.SingleAgentPlanner', return_value=mock_single_planner) as mock_planner_class:
        yield { "env": env, "predictor": predictor, "coord_manager": coord_manager, "mock_planner_class": mock_planner_class, "mock_single_planner": mock_single_planner }

@pytest.fixture
def active_mission_state():
    state = get_initial_state()
    state['drones']['Drone 1']['status'] = 'EN ROUTE'
    state['drones']['Drone 1']['mission_id'] = 'M-123'
    state['drones']['Drone 1']['pos'] = (-74.0, 40.72, 50)
    state['active_missions']['M-123'] = { 'mission_id': 'M-123', 'drone_id': 'Drone 1', 'order_ids': ['OrderX'], 'stops': [{'id': 'OrderX', 'pos': (-74.0, 40.73, 50), 'payload_kg': 1.0}], 'destinations': [(-74.0, 40.73, 50)], 'path_world_coords': [(-74.0, 40.72, 50), (-74.0, 40.73, 50)], 'start_time': 0, 'payload_kg': 1.0, 'start_battery': 200, }
    state['pending_orders'] = {}
    return state

def test_low_battery_triggers_return_to_hub(active_mission_state, mock_planners):
    state = active_mission_state
    # Energy to return is 20Wh. Threshold factor is 1.5. Required energy is 30Wh.
    mock_planners['predictor'].predict.return_value = (50.0, 20.0)
    
    # FIX: Set battery to a value (29.0) clearly below the 30Wh threshold.
    # The previous value of 30.0 failed due to a strict less-than (<) check.
    state['drones']['Drone 1']['battery'] = 29.0
    
    drone_to_check = state['drones']['Drone 1']
    contingency_triggered = check_for_contingencies(state, mock_planners, drone_to_check)
    
    assert contingency_triggered is True
    drone = state['drones']['Drone 1']
    assert drone['status'] == 'EMERGENCY_RETURN'
    assert 'M-123' not in state['active_missions']
    assert 'OrderX' in state['pending_orders']
    assert len(state['completed_missions_log']) == 1
    assert state['completed_missions_log'][0]['outcome'] == 'Failed: Critically Low Battery'
    assert drone['mission_id'].startswith('EM-')
    mock_planners['mock_single_planner'].find_strategic_path_rrt.assert_called_once()

def test_new_nfz_triggers_replanning(active_mission_state, mock_planners):
    state = active_mission_state
    env = mock_planners['env']
    env.was_nfz_just_added = True
    env.is_line_obstructed.return_value = True
    
    drone_to_check = state['drones']['Drone 1']
    contingency_triggered = check_for_contingencies(state, mock_planners, drone_to_check)
    
    assert contingency_triggered is True
    drone = state['drones']['Drone 1']
    assert drone['status'] == 'EMERGENCY_RETURN'
    assert 'M-123' not in state['active_missions']
    assert 'OrderX' in state['pending_orders']
    assert len(state['completed_missions_log']) == 1
    assert state['completed_missions_log'][0]['outcome'] == 'Failed: Path Invalidated by NFZ'
    # The flag is NOT reset by the checker itself, but by the main loop.
    # This assertion confirms the checker doesn't incorrectly reset the flag.
    assert env.was_nfz_just_added is True