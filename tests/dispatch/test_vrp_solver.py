# FILE: tests/dispatch/test_vrp_solver.py
import pytest
from unittest.mock import MagicMock
import numpy as np

from dispatch.vrp_solver import VRPSolver

@pytest.fixture
def mock_predictor():
    """A simple predictor that returns cost based on Manhattan distance."""
    predictor = MagicMock()
    def cost_func(p1, p2, *args):
        dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        return dist, dist # time, energy
    predictor.predict.side_effect = cost_func
    return predictor

@pytest.fixture
def vrp_solver(mock_predictor):
    return VRPSolver(mock_predictor)

def test_correct_batching(vrp_solver):
    """Provide 2 drones and 3 orders. Assert the solver creates two tours,
    with one drone taking two packages that are close to each other."""
    drones = [
        # FIX: Lower capacity to force a multi-drone solution.
        # Total payload is 3kg, so a capacity of 2.5kg means one drone can't take all orders.
        {'id': 'D1', 'pos': (0, 0, 10), 'max_payload_kg': 2.5},
        {'id': 'D2', 'pos': (0, 0, 10), 'max_payload_kg': 2.5}
    ]
    orders = [
        {'id': 'O1', 'pos': (10, 10, 50), 'payload_kg': 1.0}, # Far
        {'id': 'O2', 'pos': (1, 1, 50), 'payload_kg': 1.0},   # Close
        {'id': 'O3', 'pos': (2, 2, 50), 'payload_kg': 1.0}    # Close
    ]
    
    tours = vrp_solver.generate_tours(drones, orders)

    assert len(tours) == 2
    
    # Find the tour with 2 stops
    tour_with_2_stops = next((t for t in tours if len(t['stops']) == 2), None)
    assert tour_with_2_stops is not None

    # Check that the two closest orders (O2, O3) were batched together
    stop_ids = {stop['id'] for stop in tour_with_2_stops['stops']}
    assert stop_ids == {'O2', 'O3'}
    
    # Check the other tour has the single far order
    tour_with_1_stop = next((t for t in tours if len(t['stops']) == 1), None)
    assert tour_with_1_stop is not None
    assert tour_with_1_stop['stops'][0]['id'] == 'O1'


def test_payload_capacity_is_respected(vrp_solver):
    """Give one drone a 2kg capacity and two 1.5kg orders.
    Assert they are assigned to different tours."""
    drones = [
        # FIX: Set both drones to the same lower capacity.
        # This forces a split since neither drone can carry the 3kg total payload.
        {'id': 'D1', 'pos': (0, 0, 10), 'max_payload_kg': 2.0},
        {'id': 'D2', 'pos': (0, 0, 10), 'max_payload_kg': 2.0}
    ]
    orders = [
        {'id': 'O1', 'pos': (1, 1, 50), 'payload_kg': 1.5},
        {'id': 'O2', 'pos': (2, 2, 50), 'payload_kg': 1.5}
    ]

    tours = vrp_solver.generate_tours(drones, orders)

    assert len(tours) == 2
    # Each tour should have exactly one stop, as they cannot be batched
    assert all(len(t['stops']) == 1 for t in tours)
    
    # Check that the two orders were split across the two tours
    assigned_order_ids = {t['stops'][0]['id'] for t in tours}
    assert assigned_order_ids == {'O1', 'O2'}


def test_returns_no_solution_if_infeasible(vrp_solver):
    """Provide an order that is too heavy for any drone.
    Assert the solver returns an empty list."""
    drones = [
        {'id': 'D1', 'pos': (0, 0, 10), 'max_payload_kg': 2.0}
    ]
    orders = [
        {'id': 'O1', 'pos': (1, 1, 50), 'payload_kg': 3.0} # Too heavy
    ]

    tours = vrp_solver.generate_tours(drones, orders)

    # OR-Tools might return an empty list or a list with empty tours.
    # We should filter for tours that actually have stops.
    valid_tours = [t for t in tours if t['stops']]
    assert len(valid_tours) == 0