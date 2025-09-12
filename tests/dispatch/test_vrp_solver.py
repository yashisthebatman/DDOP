# FILE: tests/dispatch/test_vrp_solver.py
import pytest
from unittest.mock import MagicMock
import numpy as np

from dispatch.vrp_solver import VRPSolver
from config import HUBS

@pytest.fixture
def mock_predictor():
    """A simple predictor that returns cost based on Euclidean distance."""
    predictor = MagicMock()
    def cost_func(p1, p2, *args):
        # FIX: Use Euclidean distance for a more realistic and stable cost model in testing.
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        return dist, dist # time, energy
    predictor.predict.side_effect = cost_func
    return predictor

@pytest.fixture
def vrp_solver(mock_predictor):
    return VRPSolver(mock_predictor)

def test_correct_batching_mdvrp(vrp_solver):
    """Assert the MDVRP solver creates two tours, with one drone taking two packages that are close to each other."""
    drones = [
        {'id': 'D1', 'pos': HUBS["Hub A (South Manhattan)"], 'home_hub': "Hub A (South Manhattan)", 'max_payload_kg': 2.5},
        {'id': 'D2', 'pos': HUBS["Hub A (South Manhattan)"], 'home_hub': "Hub A (South Manhattan)", 'max_payload_kg': 2.5}
    ]
    orders = [
        {'id': 'O1', 'pos': (-73.98, 40.73, 50), 'payload_kg': 1.0},   # Far
        {'id': 'O2', 'pos': (-74.01, 40.71, 50), 'payload_kg': 1.0},  # Close
        {'id': 'O3', 'pos': (-74.01, 40.70, 50), 'payload_kg': 1.0}   # Close
    ]
    
    tours = vrp_solver.generate_tours(drones, orders)

    assert len(tours) == 2
    
    tour_with_2_stops = next((t for t in tours if len(t['stops']) == 2), None)
    assert tour_with_2_stops is not None

    stop_ids = {stop['id'] for stop in tour_with_2_stops['stops']}
    assert stop_ids == {'O2', 'O3'}
    
    tour_with_1_stop = next((t for t in tours if len(t['stops']) == 1), None)
    assert tour_with_1_stop is not None
    assert tour_with_1_stop['stops'][0]['id'] == 'O1'
    assert tour_with_1_stop['start_hub_id'] == "Hub A (South Manhattan)"


def test_payload_capacity_is_respected_mdvrp(vrp_solver):
    """Give two drones a 2kg capacity and two 1.5kg orders. Assert they are assigned to different tours."""
    drones = [
        {'id': 'D1', 'pos': HUBS["Hub A (South Manhattan)"], 'home_hub': "Hub A (South Manhattan)", 'max_payload_kg': 2.0},
        {'id': 'D2', 'pos': HUBS["Hub A (South Manhattan)"], 'home_hub': "Hub A (South Manhattan)", 'max_payload_kg': 2.0}
    ]
    orders = [
        {'id': 'O1', 'pos': (-74.0, 40.71, 50), 'payload_kg': 1.5},
        {'id': 'O2', 'pos': (-74.0, 40.72, 50), 'payload_kg': 1.5}
    ]

    tours = vrp_solver.generate_tours(drones, orders)

    assert len(tours) == 2
    assert all(len(t['stops']) == 1 for t in tours)
    
    assigned_order_ids = {t['stops'][0]['id'] for t in tours}
    assert assigned_order_ids == {'O1', 'O2'}


def test_returns_no_solution_if_infeasible_mdvrp(vrp_solver):
    """Provide an order that is too heavy for any drone. Assert the solver returns an empty list."""
    drones = [
        {'id': 'D1', 'pos': HUBS["Hub A (South Manhattan)"], 'home_hub': "Hub A (South Manhattan)", 'max_payload_kg': 2.0}
    ]
    orders = [
        {'id': 'O1', 'pos': (-74.0, 40.71, 50), 'payload_kg': 3.0} # Too heavy
    ]

    tours = vrp_solver.generate_tours(drones, orders)
    valid_tours = [t for t in tours if t['stops']]
    assert len(valid_tours) == 0