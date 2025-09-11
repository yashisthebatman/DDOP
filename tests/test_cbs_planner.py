import pytest
from unittest.mock import MagicMock
from fleet.cbs_components import Agent
from planners.cbs_planner import CBSPlanner

# A simple grid position type for tests
GridPos = tuple[int, int, int]

@pytest.fixture
def mock_low_level_planner():
    planner = MagicMock()
    
    # Store agents to recall their properties later
    planner.last_agents = []
    def find_path_mock(agent, constraints):
        planner.last_agents.append(agent)
        # Head-on conflict scenario: A(0,0)->(4,0), B(4,0)->(0,0)
        if agent.id == 1:
            path = [( (i,0,0), i ) for i in range(5)] # Path: (0,0,0) -> (4,0,0)
            if any(c.position == (2,0,0) and c.timestamp == 2 for c in constraints):
                # Detour path if constrained
                return [((0,0,0),0), ((1,0,0),1), ((1,1,0),2), ((2,1,0),3), ((3,0,0),4), ((4,0,0),5)]
            return path
        if agent.id == 2:
            path = [( (4-i,0,0), i ) for i in range(5)] # Path: (4,0,0) -> (0,0,0)
            if any(c.position == (2,0,0) and c.timestamp == 2 for c in constraints):
                # Wait path if constrained
                return [((4,0,0),0), ((3,0,0),1), ((3,0,0),2), ((2,0,0),3), ((1,0,0),4), ((0,0,0),5)]
            return path
        return []

    planner.find_path.side_effect = find_path_mock
    return planner

@pytest.fixture
def cbs_planner(mock_low_level_planner):
    env_mock = MagicMock()
    coord_manager_mock = MagicMock()
    # Mock the coordinate conversion to pass grid coords straight through
    coord_manager_mock.local_grid_to_world.side_effect = lambda p: p
    
    planner = CBSPlanner(env_mock, coord_manager_mock)
    planner.low_level_planner = mock_low_level_planner
    
    # Mock smoother to return the path as is
    smoother_mock = MagicMock()
    smoother_mock.smooth_path.side_effect = lambda path, env: path
    smoother_mock.validate_smoothed_solution.return_value = True
    planner.smoother = smoother_mock
    
    return planner

def test_finds_trivial_solution(cbs_planner):
    """Test with two agents whose paths do not conflict."""
    agent1 = Agent(id=1, start_pos=(0,0,0), goal_pos=(2,0,0), config={})
    agent2 = Agent(id=2, start_pos=(0,2,0), goal_pos=(2,2,0), config={})
    
    def find_path_no_conflict(agent, constraints):
        cbs_planner.low_level_planner.last_agents.append(agent)
        if agent.id == 1: return [((i,0,0), i) for i in range(3)]
        if agent.id == 2: return [((i,2,0), i) for i in range(3)]
        return None
    cbs_planner.low_level_planner.find_path.side_effect = find_path_no_conflict
    
    solution = cbs_planner.plan_fleet([agent1, agent2])
    assert solution is not None
    assert len(solution[1]) == 3
    assert len(solution[2]) == 3
    # Initial plan should be found without needing to replan, so find_path is called once per agent
    assert cbs_planner.low_level_planner.find_path.call_count == 2

def test_solves_head_on_conflict(cbs_planner):
    """Test the classic A->B, B->A scenario where paths collide."""
    agent1 = Agent(id=1, start_pos=(0,0,0), goal_pos=(4,0,0), config={})
    agent2 = Agent(id=2, start_pos=(4,0,0), goal_pos=(0,0,0), config={})

    solution = cbs_planner.plan_fleet([agent1, agent2])
    assert solution is not None
    
    # Check for conflict resolution
    path1 = [p for p in solution[1]]
    path2 = [p for p in solution[2]]
    
    positions1 = {p: t for p, t in zip(path1, range(len(path1)))}
    positions2 = {p: t for p, t in zip(path2, range(len(path2)))}

    # Ensure no two agents occupy the same position at the same time
    for pos, time in positions1.items():
        if pos in positions2 and positions2[pos] == time:
            pytest.fail(f"Conflict not resolved at {pos} at time {time}")