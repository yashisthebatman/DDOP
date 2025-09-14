# FILE: tests/test_multi_depot.py

import pytest
from unittest.mock import MagicMock
import numpy as np

from server import update_simulation
from system_state import get_initial_state
from config import HUBS
from utils.coordinate_manager import CoordinateManager

# Helper to get hub by ID from the new config format
def get_hub_by_id(hub_id):
    return next((hub for hub in HUBS if hub['id'] == hub_id), None)

@pytest.fixture
def mock_planners():
    """Provides a mock planners dict needed by the real update_simulation."""
    return {"coord_manager": CoordinateManager()}

def test_drone_relocates_after_rebalance_mission(mock_planners):
    """Simulate a rebalancing mission and assert drone's home hub is updated."""
    state = get_initial_state()
    drone_id = "Drone 1" # From HUB_A
    drone = state['drones'][drone_id]
    
    drone['status'] = 'RETURNING_TO_HUB' # Final leg of mission
    drone['mission_id'] = 'M-REBALANCE'
    drone['home_hub'] = 'HUB_A'
    
    end_hub_obj = get_hub_by_id('HUB_B')
    
    mission = {
        'mission_id': 'M-REBALANCE', 'drone_id': drone_id, 'order_ids': [], 
        'destinations': [end_hub_obj['location']], 'start_time': 0.0, 
        'total_planned_time': 200.0, 'path_world_coords': [drone['pos'], end_hub_obj['location']],
        'start_hub': 'HUB_A', 'end_hub': 'HUB_B', 'stops': [],
        'start_battery': 200, 'total_planned_energy': 30, 
        'mission_time_elapsed': 0.0, 'flight_time_elapsed': 0.0, 'total_maneuver_time': 0,
        'current_path_target_index': 1, 'current_stop_index': 0,
    }
    state['active_missions']['M-REBALANCE'] = mission
    
    loop_count = 0
    while 'M-REBALANCE' in state['active_missions'] and loop_count < 1000:
        update_simulation(state, mock_planners)
        loop_count += 1
    
    assert loop_count < 1000, "Simulation timed out"
    assert drone['status'] == 'RECHARGING'
    assert drone['home_hub'] == 'HUB_B'
    assert drone['pos'] == end_hub_obj['location']