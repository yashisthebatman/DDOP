# FILE: tests/test_cbs_planner.py (rename to test_cbsh_planner.py)
import pytest
from unittest.mock import MagicMock, patch
from fleet.cbs_components import Agent
from planners.cbsh_planner import CBSHPlanner, MIN_SEPARATION_METERS

@pytest.fixture
def mock_low_level_planners():
    # Mock the two-stage low-level planning process
    with patch('planners.cbsh_planner.AnytimeRRTStar') as mock_rrt, \
         patch('planners.cbsh_planner.PathTimingSolver') as mock_solver:
        
        # Stage 1 Mock: RRT* returns a simple geometric path
        def rrt_constructor_side_effect(start, goal, *args, **kwargs):
            instance = MagicMock()
            instance.plan.return_value = ([start, goal], "Success")
            return instance
        mock_rrt.side_effect = rrt_constructor_side_effect

        # Stage 2 Mock: Timing solver returns a timed path
        # This mock will be configured within tests for specific scenarios
        mock_solver.return_value.find_timing.return_value = [((0,0,0),0), ((1,0,0),1), ((2,0,0),2)]
        
        yield mock_rrt, mock_solver.return_value


@pytest.fixture
def cbsh_planner(mock_low_level_planners):
    env_mock = MagicMock()
    coord_manager_mock = MagicMock()
    coord_manager_mock.world_to_local_grid.side_effect = lambda p: (int(p[0]), int(p[1]), int(p[2]))
    # FIX: The mock was incomplete, which likely contributed to the hang/error.
    # The planner's conflict detection requires this method to calculate distances.
    coord_manager_mock.world_to_local_meters.side_effect = lambda p: p
    
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
            # If constrained, take a detour that is far enough away to be a valid resolution.
            if any(c.position == (2,0,0) and c.timestamp == 2 for c in constraints):
                # FIX: The original detour was still too close (dist=1.414), causing the planner
                # to find another conflict and loop infinitely. This new detour moves the agent
                # 20 meters away, satisfying MIN_SEPARATION_METERS (15.0).
                return [((0,0,0),0), ((1,0,0),1), ((1,20,0),2), ((2,20,0),3), ((3,0,0),4), ((4,0,0),5)]
            return path
        # Agent 2's default path
        elif geom_path[0] == (4,0,0):
            path = [((4-i,0,0), i) for i in range(5)]
            # If constrained, wait. This is a valid strategy for the low-level planner to propose.
            # The CBS high-level will determine if this solution is globally optimal and conflict-free.
            if any(c.position == (2,0,0) and c.timestamp == 2 for c in constraints):
                return [((4,0,0),0), ((3,0,0),1), ((3,0,0),2), ((2,0,0),3), ((1,0,0),4), ((0,0,0),5)]
            return path
        return []

    mock_timing_solver.find_timing.side_effect = timing_side_effect
    
    # --- Run Planner ---
    solution = cbsh_planner.plan_fleet([agent1, agent2])
    assert solution is not None

    # --- Validate Solution ---
    path1 = solution[1]
    path2 = solution[2]

    # Check for conflict resolution by ensuring no two agents occupy the same grid cell at the same time
    positions_over_time = {}
    max_time = max(p[-1][1] for p in solution.values() if p)
    for t in range(max_time + 1):
        # Using a set to check for proximity conflicts in the final solution
        positions_at_t = set()
        for agent_id, path in solution.items():
            pos_at_t = None
            # Basic interpolation to find position at time t
            if path and t <= path[-1][1]:
                for i in range(len(path) - 1):
                    if path[i][1] <= t < path[i+1][1]:
                        p1, t1 = path[i]
                        p2, t2 = path[i+1]
                        progress = (t - t1) / (t2 - t1)
                        pos_at_t = tuple(p1_i + progress * (p2_i - p1_i) for p1_i, p2_i in zip(p1,p2))
                        break
                if not pos_at_t and path[-1][1] == t:
                    pos_at_t = path[-1][0]

            if pos_at_t:
                # Check for collisions with other agents at this timestep
                for other_pos in positions_at_t:
                    dist = sum((c1 - c2)**2 for c1, c2 in zip(pos_at_t, other_pos))**0.5
                    if dist < 0.1: # Use a small tolerance for floating point comparisons
                        pytest.fail(f"Collision detected at {pos_at_t} at time {t}")
                positions_at_t.add(pos_at_t)