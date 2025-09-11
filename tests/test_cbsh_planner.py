# FILE: tests/test_cbs_planner.py (rename to test_cbsh_planner.py)
import pytest
from unittest.mock import MagicMock, patch
from fleet.cbs_components import Agent
from planners.cbsh_planner import CBSHPlanner

@pytest.fixture
def mock_low_level_planners():
    # Mock the two-stage low-level planning process
    with patch('planners.cbsh_planner.AnytimeRRTStar') as mock_rrt, \
         patch('planners.cbsh_planner.PathTimingSolver') as mock_solver:
        
        # Stage 1 Mock: RRT* returns a simple geometric path
        mock_rrt.return_value.plan.return_value = ([(0,0,0), (1,0,0), (2,0,0)], "Success")

        # Stage 2 Mock: Timing solver returns a timed path
        # This mock will be configured within tests for specific scenarios
        mock_solver.return_value.find_timing.return_value = [((0,0,0),0), ((1,0,0),1), ((2,0,0),2)]
        
        yield mock_rrt, mock_solver.return_value


@pytest.fixture
def cbsh_planner(mock_low_level_planners):
    env_mock = MagicMock()
    coord_manager_mock = MagicMock()
    coord_manager_mock.world_to_local_grid.side_effect = lambda p: (int(p[0]), int(p[1]), int(p[2]))

    planner = CBSHPlanner(env_mock, coord_manager_mock)
    return planner

def test_solves_head_on_conflict(cbsh_planner, mock_low_level_planners):
    """Test the classic A->B, B->A scenario where paths collide."""
    mock_rrt, mock_timing_solver = mock_low_level_planners

    agent1 = Agent(id=1, start_pos=(0,0,0), goal_pos=(4,0,0), config={})
    agent2 = Agent(id=2, start_pos=(4,0,0), goal_pos=(0,0,0), config={})

    # --- Configure Mocks for Conflict Scenario ---
    def timing_side_effect(geom_path, constraints):
        # Agent 1's default path
        if geom_path[0] == (0,0,0): 
            path = [((i,0,0), i) for i in range(5)]
            # If constrained at the conflict point, detour
            if any(c.position == (2,0,0) and c.timestamp == 2 for c in constraints):
                return [((0,0,0),0), ((1,0,0),1), ((1,1,0),2), ((2,1,0),3), ((3,0,0),4), ((4,0,0),5)]
            return path
        # Agent 2's default path
        elif geom_path[0] == (4,0,0):
            path = [((4-i,0,0), i) for i in range(5)]
            # If constrained, wait
            if any(c.position == (2,0,0) and c.timestamp == 2 for c in constraints):
                return [((4,0,0),0), ((3,0,0),1), ((3,0,0),2), ((2,0,0),3), ((1,0,0),4), ((0,0,0),5)]
            return path
        return []

    mock_timing_solver.find_timing.side_effect = timing_side_effect
    
    # RRT just needs to return a plausible geometric path for each agent
    def rrt_side_effect(start, goal, *args, **kwargs):
        return ([start, goal], "Success")
    mock_rrt.return_value.plan.side_effect = rrt_side_effect

    # --- Run Planner ---
    solution = cbsh_planner.plan_fleet([agent1, agent2])
    assert solution is not None

    # --- Validate Solution ---
    path1 = solution[1]
    path2 = solution[2]

    # Check for conflict resolution by ensuring no two agents occupy the same grid cell at the same time
    positions_over_time = {}
    for t in range(10): # Check up to a reasonable time
        for agent_id, path in solution.items():
            pos_at_t = None
            if path and t >= path[0][1] and t <= path[-1][1]:
                # Find waypoint for time t
                for i in range(len(path) - 1):
                    if path[i][1] <= t < path[i+1][1]:
                        pos_at_t = cbsh_planner.coord_manager.world_to_local_grid(path[i][0])
                        break
                if not pos_at_t and path[-1][1] == t:
                    pos_at_t = cbsh_planner.coord_manager.world_to_local_grid(path[-1][0])
            
            if pos_at_t:
                if (pos_at_t, t) in positions_over_time:
                    pytest.fail(f"Conflict not resolved at {pos_at_t} at time {t}")
                positions_over_time[(pos_at_t, t)] = agent_id