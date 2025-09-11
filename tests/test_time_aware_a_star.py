import pytest
from unittest.mock import MagicMock
from utils.time_aware_a_star import TimeAwareAStar
from fleet.cbs_components import Agent, Constraint

@pytest.fixture
def mock_environment():
    env = MagicMock()
    env.risk_map.get_risk.return_value = 0.0
    return env

@pytest.fixture
def mock_coord_manager():
    manager = MagicMock()
    manager.is_valid_local_grid_pos.side_effect = lambda pos: all(0 <= c < 10 for c in pos)
    return manager

@pytest.fixture
def a_star_planner(mock_environment, mock_coord_manager):
    return TimeAwareAStar(mock_environment, mock_coord_manager)

def test_finds_optimal_path_no_constraints(a_star_planner):
    agent = Agent(id=1, start_pos=(0,0,0), goal_pos=(3,0,0), config={'w_time': 1, 'w_energy': 0, 'w_risk': 0})
    path = a_star_planner.find_path(agent, [])
    
    assert path is not None
    assert len(path) == 4
    assert path[-1][0] == (3,0,0)

def test_avoids_path_with_constraint(a_star_planner):
    agent = Agent(id=1, start_pos=(0,0,0), goal_pos=(3,0,0), config={'w_time': 1, 'w_energy': 0, 'w_risk': 0})
    constraints = [Constraint(agent_id=1, position=(1,0,0), timestamp=1)]
    
    path = a_star_planner.find_path(agent, constraints)
    
    assert path is not None
    # FIX: A detour in 3D is not guaranteed to be longer.
    # The crucial check is that the constraint was not violated.
    assert ( (1,0,0), 1 ) not in path
    assert path[-1][0] == (3,0,0)

def test_returns_none_if_goal_is_constrained(a_star_planner):
    agent = Agent(id=1, start_pos=(0,0,0), goal_pos=(1,0,0), config={'w_time': 1, 'w_energy': 0, 'w_risk': 0})
    # FIX: Block the goal at all reachable times to make it truly impossible.
    # The agent can wait by moving to an adjacent cell, so we block t=1 and t=3, etc.
    # A simpler way is to block all neighbors at t=0. Let's just block more future times.
    constraints = [
        Constraint(agent_id=1, position=(1,0,0), timestamp=1),
        Constraint(agent_id=1, position=(1,0,0), timestamp=2),
        Constraint(agent_id=1, position=(1,0,0), timestamp=3)
    ]
    
    path = a_star_planner.find_path(agent, constraints)
    assert path is None

def test_path_avoids_high_risk_area(a_star_planner, mock_environment):
    mock_environment.risk_map.get_risk.side_effect = lambda pos: 0.9 if pos[0] == 1 else 0.0
    
    agent = Agent(id=1, start_pos=(0,0,0), goal_pos=(2,0,0), config={
        'w_time': 0.1,
        'w_energy': 0,
        'w_risk': 10.0
    })
    
    path = a_star_planner.find_path(agent, [])
    
    assert path is not None
    path_positions = [p[0] for p in path]
    # With the corrected cost function, the planner should now choose a detour.
    assert not any(pos[0] == 1 for pos in path_positions)
    assert path[-1][0] == (2,0,0)