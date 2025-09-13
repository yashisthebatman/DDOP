# FILE: tests/test_multi_depot.py

import pytest
from unittest.mock import MagicMock
import numpy as np
import time

# FIX: Import the REAL simulation logic, not a local copy.
from server import update_simulation
from system_state import get_initial_state
from config import HUBS, DRONE_BATTERY_WH, DRONE_RECHARGE_TIME_S
from dispatch.vrp_solver import VRPSolver

@pytest.fixture
def mock_predictor():
    """A simple predictor that returns cost based on Euclidean distance."""
    predictor = MagicMock()
    def cost_func(p1, p2, *args):
        dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
        return dist, dist
    predictor.predict.side_effect = cost_func
    return predictor

@pytest.fixture
def mock_planners():
    """Provides a mock planners dict needed by the real update_simulation."""
    return {"coord_manager": MagicMock()}

def test_vrp_selects_closest_hub_for_end(mock_predictor):
    """Assert solver chooses the closest hub to the last delivery as the end point."""
    vrp_solver = VRPSolver(mock_predictor)
    
    drones = [{'id': 'D1', 'pos': HUBS['Hub A (South Manhattan)'], 'home_hub': 'Hub A (South Manhattan)', 'max_payload_kg': 5.0}]
    # This order is physically very close to Hub C
    orders = [{'id': 'O1', 'pos': (-74.007, 40.736, 50.0), 'payload_kg': 1.0}]
    
    tours = vrp_solver.generate_tours(drones, orders)
    
    assert len(tours) == 1
    tour = tours[0]
    # The mission should start at the drone's home hub and end at the closest hub to the delivery
    assert tour['start_hub_id'] == 'Hub A (South Manhattan)'
    assert tour['end_hub_id'] == 'Hub C (West Side)'

def test_drone_relocates_after_mission(mock_planners):
    """Simulate a mission and assert drone's home hub is updated on completion."""
    state = get_initial_state()
    # Ensure at least one drone exists for the test
    if not state['drones']:
        pytest.fail("Initial state has no drones.")
    
    drone_id = list(state['drones'].keys())[0]
    drone = state['drones'][drone_id]
    
    drone['status'] = 'EN ROUTE'
    drone['mission_id'] = 'M-ABC'
    drone['home_hub'] = 'Hub A (South Manhattan)'
    
    mission = {
        'mission_id': 'M-ABC', 'drone_id': drone_id, 'order_ids': ['O1'], 
        'destinations': [HUBS['Hub B (Midtown East)']], 'start_time': 0.0, 
        'total_planned_time': 50.0, 'path_world_coords': [(0,0,0), (1,1,1)],
        'start_hub': 'Hub A (South Manhattan)', 'end_hub': 'Hub B (Midtown East)',
        'start_battery': 200, 'total_planned_energy': 30, 'stops': [],
        'mission_time_elapsed': 49.5, 'flight_time_elapsed': 49.5, 'total_maneuver_time': 0
    }
    state['active_missions']['M-ABC'] = mission
    
    # Fast-forward time to complete the mission on the next tick
    state['simulation_time'] = 50.0
    update_simulation(state, mock_planners) # FIX: Pass planners dict
    
    assert drone['status'] == 'RECHARGING'
    assert drone['home_hub'] == 'Hub B (Midtown East)'
    assert drone['pos'] == HUBS['Hub B (Midtown East)']