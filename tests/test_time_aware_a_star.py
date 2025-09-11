import pytest
from unittest.mock import MagicMock
from itertools import product
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
    # Allow a slightly larger grid for the inescapable box test
    manager.is_valid_local_grid_pos.side_effect = lambda pos: all(-2 <= c < 10 for c in pos)
    manager.local_grid_to_world.side_effect = lambda p: p
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
    assert ( (1,0,0), 1 ) not in path
    assert path[-1][0] == (3,0,0)

def test_returns_none_if_goal_is_constrained(a_star_planner):
    agent = Agent(id=1, start_pos=(0,0,0), goal_pos=(2,0,0), config={'w_time': 1, 'w_energy': 0, 'w_risk': 0})
    
    # FIX: Create a truly inescapable box. At t=1, every single cell adjacent
    # to the start position (including the start position itself) is constrained.
    # The agent has nowhere to move from t=0 to t=1, making a path impossible.
    # This is a robust test of unsolvable scenarios.
    constraints = []
    start_pos = agent.start_pos
    moves = [move for move in product([-1, 0, 1], repeat=3)]
    for move in moves:
        constrained_pos = (start_pos[0] + move[0], start_pos[1] + move[1], start_pos[2] + move[2])
        constraints.append(Constraint(agent_id=1, position=constrained_pos, timestamp=1))

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
    assert not any(pos[0] == 1 for pos in path_positions)
    assert path[-1][0] == (2,0,0)