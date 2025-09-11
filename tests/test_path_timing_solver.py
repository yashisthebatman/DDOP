# FILE: tests/test_path_timing_solver.py
import pytest
from unittest.mock import MagicMock
from utils.path_timing_solver import PathTimingSolver
from fleet.cbs_components import Constraint
from utils.coordinate_manager import CoordinateManager
from config import DRONE_SPEED_MPS

@pytest.fixture
def mock_coord_manager():
    manager = MagicMock(spec=CoordinateManager)
    # Mock a simple conversion: 1 unit of world coord = 1000 meters
    manager.world_to_local_meters.side_effect = lambda p: (p[0]*1000, p[1]*1000, p[2])
    manager.world_to_local_grid.side_effect = lambda p: (int(p[0]), int(p[1]), int(p[2]))
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
    assert timed_path[0] == ((0,0,10), 0)
    assert timed_path[1] == ((0.02,0,10), 1)
    assert timed_path[2] == ((0.04,0,10), 2)

def test_timing_with_wait_constraint(solver):
    """Tests if the solver can wait at a waypoint to avoid a constraint."""
    geom_path = [(0,0,10), (1,0,10), (2,0,10)]
    # Constraint: Cannot be at grid pos (1,0,10) at time=1
    constraints = [Constraint(agent_id=1, position=(1,0,10), timestamp=1)]
    
    timed_path = solver.find_timing(geom_path, constraints)

    assert timed_path is not None
    # Check that the drone is NOT at (1,0,10) at time=1
    # It should have waited at (0,0,10) for t=1 and arrived at (1,0,10) later.
    positions_at_time_1 = [pos for pos, t in timed_path if t == 1]
    assert (1,0,10) not in positions_at_time_1

def test_timing_fails_if_unsolvable(solver):
    """Tests that the solver returns None if the path is impossible."""
    geom_path = [(0,0,10), (1,0,10)]
    # Constraint: The goal itself is blocked at all early times
    constraints = [Constraint(1, (1,0,10), t) for t in range(1, 15)]
    
    timed_path = solver.find_timing(geom_path, constraints)
    assert timed_path is None