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
        'battery': 100.0 # Start with plenty of battery
    })
    
    state['active_missions'][mission_id] = {
        'mission_id': mission_id,
        'drone_id': drone_id,
        'order_ids': ['OrderY'],
        'stops': [{'id': 'OrderY', 'pos': (-74.0, 40.73, 50), 'payload_kg': 1.0}],
        'destinations': [(-74.0, 40.73, 50), hub_a_loc], # Delivery and return
        'start_time': 0,
        'payload_kg': 1.0,
        'start_battery': 200.0
    }
    state['pending_orders'] = {}
    return state

def test_in_flight_low_battery_triggers_emergency_return(active_mission_state, mock_planners):
    """
    Verify that if a drone's battery drops below the safety threshold mid-flight,
    the contingency planner correctly triggers an emergency return.
    """
    state = active_mission_state
    
    # Mock predictor to report energy costs that will fail the check.
    # Energy to finish mission = 30Wh. Energy to return after = 20Wh.
    # Total required with RTH_FACTOR=1.5 is (30 + 20) * 1.5 = 75Wh.
    mock_planners['predictor'].predict.side_effect = [
        (100.0, 30.0), # For predict(current -> final_dest)
        (80.0, 20.0)   # For predict(final_dest -> nearest_hub)
    ]
    
    # Set drone's battery to a value that is NOT enough to meet the safety margin.
    state['drones']['Drone 1']['battery'] = 74.0
    
    # Run the check
    check_for_contingencies(state, mock_planners)
    
    # Assertions
    drone = state['drones']['Drone 1']
    assert drone['status'] == 'EMERGENCY_RETURN'
    assert 'M-TEST-BATT' not in state['active_missions'] # Original mission aborted
    assert 'OrderY' in state['pending_orders'] # Order returned to queue
    assert len(state['completed_missions_log']) == 1
    assert state['completed_missions_log'][0]['outcome'] == 'Failed: Critically Low Battery'
    assert drone['mission_id'].startswith('EM-')