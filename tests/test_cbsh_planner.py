# FILE: tests/test_cbsh_planner.py
import pytest
from unittest.mock import MagicMock, patch
from fleet.cbs_components import Agent, Conflict, Constraint
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
    # Mock grid conversion to simple integer truncation
    coord_manager_mock.world_to_local_grid.side_effect = lambda p: (int(p[0]), int(p[1]), int(p[2]))
    coord_manager_mock.world_to_local_meters.side_effect = lambda p: p
    
    planner = CBSHPlanner(env_mock, coord_manager_mock)
    return planner

def test_solves_head_on_conflict(cbsh_planner, mock_low_level_planners):
    """Test the classic A->B, B->A scenario where paths collide."""
    mock_rrt, mock_timing_solver = mock_low_level_planners

    agent1 = Agent(id=1, start_pos=(0,0,0), goal_pos=(4,0,0), config={})
    agent2 = Agent(id=2, start_pos=(4,0,0), goal_pos=(0,0,0), config={})

    # --- Define Default (conflicting) and Alternative (safe) Paths ---
    path1_default = [((i,0,0), i) for i in range(5)]
    path2_default = [((4-i,0,0), i) for i in range(5)]
    path_alt_detour = [((0,0,0),0), ((1,0,0),1), ((1,20,0),2), ((2,20,0),3), ((3,0,0),4), ((4,0,0),5)]
    path_alt_wait = [((4,0,0),0), ((3,0,0),1), ((3,0,0),2), ((2,0,0),3), ((1,0,0),4), ((0,0,0),5)]

    # --- FIX: Create an intelligent mock that behaves like a real low-level planner ---
    # This new mock checks if its default path violates any given constraints. If so,
    # it returns a pre-defined safe alternative. This prevents the infinite loop.
    def timing_side_effect(geom_path, constraints):
        # Determine which agent is being planned for based on its start position
        agent_id = 1 if geom_path[0] == agent1.start_pos else 2
        
        default_path = path1_default if agent_id == 1 else path2_default
        alt_path = path_alt_detour if agent_id == 1 else path_alt_wait

        is_violated = False
        # Check if the default path violates any constraints specific to this agent
        for c in constraints:
            if c.agent_id == agent_id:
                for pos, time in default_path:
                    grid_pos = cbsh_planner.coord_manager.world_to_local_grid(pos)
                    if grid_pos == c.position and time == c.timestamp:
                        is_violated = True
                        break
            if is_violated:
                break
        
        return alt_path if is_violated else default_path

    mock_timing_solver.find_timing.side_effect = timing_side_effect
    
    # --- Run Planner ---
    solution = cbsh_planner.plan_fleet([agent1, agent2])
    assert solution is not None

    # --- Validate Solution ---
    # Check for conflict resolution by ensuring no two agents occupy the same point at the same time
    max_time = max(p[-1][1] for p in solution.values() if p)
    for t in range(max_time + 1):
        positions_at_t = set()
        for agent_id, path in solution.items():
            pos_at_t = None
            if path and t <= path[-1][1]:
                if len(path) == 1 and path[0][1] == t:
                    pos_at_t = path[0][0]
                else:
                    for i in range(len(path) - 1):
                        if path[i][1] <= t < path[i+1][1]:
                            p1, t1 = path[i]
                            p2, t2 = path[i+1]
                            progress = (t - t1) / (t2 - t1) if (t2 - t1) > 0 else 0
                            pos_at_t = tuple(p1_i + progress * (p2_i - p1_i) for p1_i, p2_i in zip(p1,p2))
                            break
                if not pos_at_t and path[-1][1] == t:
                    pos_at_t = path[-1][0]

            if pos_at_t:
                # Use a small tolerance for floating point comparisons
                rounded_pos = tuple(round(c, 3) for c in pos_at_t)
                if rounded_pos in positions_at_t:
                     pytest.fail(f"Collision detected at {rounded_pos} at time {t}")
                positions_at_t.add(rounded_pos)