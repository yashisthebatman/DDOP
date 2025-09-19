# FILE: tests/simulation/test_contingency_battery.py
import pytest
from unittest.mock import MagicMock, patch

from simulation.contingency_planner import check_for_contingencies
from system_state import get_initial_state
from config import HUBS, RTH_BATTERY_THRESHOLD_FACTOR

@pytest.fixture
def mock_planners():
    env = MagicMock()
    env.was_nfz_just_added = False
    
    predictor = MagicMock()
    coord_manager = MagicMock()
    coord_manager.world_to_meters.side_effect = lambda p: (p[0] * 1000, p[1] * 1000, p[2])
    
    mock_single_planner = MagicMock()
    mock_single_planner.find_strategic_path_rrt.return_value = ([(-74.0, 40.7, 50), (-74.0, 40.71, 50)], "Success")
    
    with patch('simulation.contingency_planner.SingleAgentPlanner', return_value=mock_single_planner):
        yield {
            "env": env,
            "predictor": predictor,
            "coord_manager": coord_manager
        }

@pytest.fixture
def active_mission_state():
    state = get_initial_state()
    drone_id = "Drone 1"
    mission_id = "M-TEST-BATT"
    
    hub_a_loc = next(h['location'] for h in HUBS if h['id'] == 'HUB_A')
    
    state['drones'][drone_id].update({
        'status': 'EN ROUTE',
        'mission_id': mission_id,
        'pos': (-74.0, 40.72, 50),
        'battery': 100.0
    })
    
    state['active_missions'][mission_id] = {
        'mission_id': mission_id,
        'drone_id': drone_id,
        'order_ids': ['OrderY'],
        'stops': [{'id': 'OrderY', 'pos': (-74.0, 40.73, 50), 'payload_kg': 1.0}],
        'destinations': [(-74.0, 40.73, 50), hub_a_loc],
        'start_time': 0,
        'payload_kg': 1.0,
        'start_battery': 200.0
    }
    state['pending_orders'] = {}
    return state

def test_in_flight_low_battery_triggers_emergency_return(active_mission_state, mock_planners):
    state = active_mission_state
    
    # Energy to return is 30Wh. Threshold factor is 1.5. Required energy is 45Wh.
    mock_planners['predictor'].predict.return_value = (100.0, 30.0)
    
    # FIX: Set battery to a value (44.0) that is clearly below the 45Wh threshold.
    # The previous value (74.0) was too high to trigger the contingency.
    state['drones']['Drone 1']['battery'] = 44.0
    
    # FIX: Call the function correctly, passing the specific drone.
    drone_to_check = state['drones']['Drone 1']
    contingency_triggered = check_for_contingencies(state, mock_planners, drone_to_check)
    
    assert contingency_triggered is True
    drone = state['drones']['Drone 1']
    assert drone['status'] == 'EMERGENCY_RETURN'
    assert 'M-TEST-BATT' not in state['active_missions']
    assert 'OrderY' in state['pending_orders']
    assert len(state['completed_missions_log']) == 1
    assert state['completed_missions_log'][0]['outcome'] == 'Failed: Critically Low Battery'
    assert drone['mission_id'].startswith('EM-')