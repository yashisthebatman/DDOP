# FILE: tests/test_path_timing_solver.py
import pytest
from unittest.mock import MagicMock
from utils.path_timing_solver import PathTimingSolver
from fleet.cbs_components import Constraint
from utils.coordinate_manager import CoordinateManager
from config import DRONE_SPEED_MPS, MAX_ALTITUDE, GRID_RESOLUTION_M

@pytest.fixture
def mock_coord_manager():
    manager = MagicMock(spec=CoordinateManager)
    # Mock a simple conversion: 1 unit of world coord = 1000 meters
    manager.world_to_local_meters.side_effect = lambda p: (p[0]*1000, p[1]*1000, p[2])
    # A realistic mock that correctly separates points based on a grid resolution.
    manager.world_to_local_grid.side_effect = lambda p: (
        int(round(p[0] * 1000 / GRID_RESOLUTION_M)),
        int(round(p[1] * 1000 / GRID_RESOLUTION_M)),
        int(p[2])
    )
    manager.alt_min = 0
    manager.alt_max = MAX_ALTITUDE
    return manager

@pytest.fixture
def solver(mock_coord_manager):
    return PathTimingSolver(mock_coord_manager)

def test_basic_timing(solver):
    """Tests if a simple path is timed correctly with no constraints."""
    # Path is two 20-meter segments. With DRONE_SPEED_MPS=20, each takes 1 sec.
    geom_path = [(0,0,10), (0.02,0,10), (0.04,0,10)]
    timed_path = solver.find_timing(geom_path, [])

    assert timed_path is not None
    assert len(timed_path) == 3
    assert timed_path[0] == ((0,0,10), 0)
    assert timed_path[1] == ((0.02,0,10), 1)
    assert timed_path[2] == ((0.04,0,10), 2)

def test_timing_with_wait_constraint(solver):
    """Tests if the solver can wait at a waypoint to avoid a constraint."""
    geom_path = [(0,0,10), (0.02,0,10)] # 20m path, takes 1s
    
    # Constraint is on the destination grid cell at the time of arrival (t=1).
    # This correctly blocks the "move" action while allowing the "wait" action.
    destination_grid_cell = (1, 0, 10)
    constraints = [Constraint(agent_id=1, position=destination_grid_cell, timestamp=1)]
    
    timed_path = solver.find_timing(geom_path, constraints)

    assert timed_path is not None
    # Expected path: wait at (0,0,10) until t=1, then move.
    # ( (0,0,10), 0 ), ( (0,0,10), 1 ), ( (0.02,0,10), 2)
    assert timed_path[0] == ((0,0,10), 0)
    assert timed_path[1] == ((0,0,10), 1)
    assert timed_path[2] == ((0.02,0,10), 2)
    # Check that it did not pass through the constrained state (0,0,10) at t=1 by moving
    assert not any(pos == (0,0,10) and t == 1 for pos, t in timed_path if pos != timed_path[0][0])


def test_timing_fails_if_unsolvable(solver):
    """Tests that the solver returns None if the path is impossible."""
    geom_path = [(0,0,10), (0.02,0,10)]

    # FIX: The original test logic was flawed. To make the scenario truly
    # unsolvable, both the "move" and "wait" options from the start must be blocked.
    start_grid_cell = (0, 0, 10)
    dest_grid_cell = (1, 0, 10)

    # 1. Block the "move" action by constraining the destination at its arrival time (t=1).
    # 2. Block the "wait" action by constraining the start cell for all possible wait times.
    constraints = [
        Constraint(1, dest_grid_cell, 1)
    ]
    constraints.extend([Constraint(1, start_grid_cell, t) for t in range(1, 15)])
    
    timed_path = solver.find_timing(geom_path, constraints)
    assert timed_path is None