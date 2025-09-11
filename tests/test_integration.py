# FILE: tests/test_integration.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# --- Import all the real components we need to integrate ---
from environment import Environment
from utils.coordinate_manager import CoordinateManager
from utils.path_smoother import PathSmoother
from ml_predictor.predictor import EnergyTimePredictor
from planners.cbsh_planner import CBSHPlanner, MIN_SEPARATION_METERS
from fleet.manager import FleetManager, Mission
from fleet.cbs_components import Agent


# --- Fixtures to provide real, shared instances of our components ---

@pytest.fixture(scope="module")
def real_coord_manager():
    """Provides a single, real CoordinateManager instance for all tests in this module."""
    return CoordinateManager()

@pytest.fixture(scope="module")
def real_environment(real_coord_manager):
    """Provides a single, real Environment instance, correctly initialized."""
    from environment import WeatherSystem
    return Environment(WeatherSystem(seed=123), real_coord_manager)

@pytest.fixture
def real_cbsh_planner(real_environment, real_coord_manager):
    """Provides a new, real CBSHPlanner instance for each test."""
    return CBSHPlanner(real_environment, real_coord_manager)


# --- Happy Path Integration Tests ---

def test_full_planning_stack_solves_head_on_conflict(real_cbsh_planner, real_coord_manager):
    """
    Tests the entire planner stack (CBSH -> RRT* -> TimingSolver) against a
    real environment to solve a head-on conflict. This is a deep integration test.
    """
    agent1 = Agent(id="DroneA", start_pos=(-74.01, 40.73, 50), goal_pos=(-73.99, 40.73, 50), config={})
    agent2 = Agent(id="DroneB", start_pos=(-73.99, 40.73, 50), goal_pos=(-74.01, 40.73, 50), config={})
    
    solution = real_cbsh_planner.plan_fleet([agent1, agent2])

    assert solution is not None, "Planner should have found a solution."
    assert "DroneA" in solution and solution["DroneA"] is not None
    assert "DroneB" in solution and solution["DroneB"] is not None

    # Rigorous check: Interpolate paths and ensure minimum separation is never violated.
    path_a = real_cbsh_planner._get_interpolated_path(solution["DroneA"])
    path_b = real_cbsh_planner._get_interpolated_path(solution["DroneB"])
    max_time = min(len(path_a), len(path_b))

    for t in range(max_time):
        pos_a = path_a[t]
        pos_b = path_b[t]
        if pos_a is not None and pos_b is not None:
            pos_a_m = real_coord_manager.world_to_local_meters(pos_a)
            pos_b_m = real_coord_manager.world_to_local_meters(pos_b)
            distance = np.linalg.norm(np.array(pos_a_m) - np.array(pos_b_m))
            assert distance > MIN_SEPARATION_METERS, f"Conflict at time {t}! Drones are {distance:.2f}m apart."


def test_fleet_manager_updates_missions_on_success(real_environment, real_coord_manager):
    """
    Tests the FleetManager's logic for orchestrating a planning cycle.
    """
    mock_planner = CBSHPlanner(real_environment, real_coord_manager)
    mock_solution = {
        "Drone1": [((-74.0, 40.7, 50), 0), ((-74.0, 40.7, 60), 10)],
        "Drone2": [((-73.9, 40.8, 50), 0), ((-73.9, 40.8, 60), 12)],
    }
    mission1 = Mission("Drone1", (-74.0, 40.7, 50), [(-74.0, 40.7, 60)], 1.0, {})
    mission2 = Mission("Drone2", (-73.9, 40.8, 50), [(-73.9, 40.8, 60)], 2.0, {})
    predictor = EnergyTimePredictor()
    predictor.fallback_predictor.predict = lambda *args, **kwargs: (10, 25)
    fm = FleetManager(mock_planner, predictor)
    fm.add_mission(mission1)
    fm.add_mission(mission2)

    with patch.object(mock_planner, 'plan_fleet', return_value=mock_solution) as mock_plan_method:
        success = fm.execute_planning_cycle()

        assert success is True
        mock_plan_method.assert_called_once()
        assert fm.missions["Drone1"].state == "IN_PROGRESS"
        assert fm.missions["Drone1"].total_planned_time == 10


# --- Boundary and Edge Case Integration Tests ---

def test_planner_fails_gracefully_for_obstructed_goal(real_cbsh_planner):
    """
    Edge Case: Tests that the low-level planner (RRT*) returns None if the goal
    is inside a No-Fly Zone, and CBSH handles this failure correctly.
    """
    # ARRANGE: Define a goal that is known to be inside a static NFZ
    # From config.py: NO_FLY_ZONES = [[-74.01, 40.715, -73.995, 40.725], ...]
    obstructed_goal = (-74.00, 40.72, 50.0)
    agent = Agent(id="DroneC", start_pos=(-74.02, 40.72, 50), goal_pos=obstructed_goal, config={})

    # ACT: Attempt to plan the impossible mission
    solution = real_cbsh_planner.plan_fleet([agent])

    # ASSERT: The planner should fail and return None, not crash.
    assert solution is None

@pytest.mark.slow  # Mark as a slow test
def test_cbsh_terminates_for_unsolvable_conflict(real_environment, real_coord_manager):
    """
    Edge Case: Tests that the high-level CBSH planner gives up on a provably
    unsolvable problem (two drones in a narrow corridor) instead of looping forever.
    """
    # ARRANGE: We simulate a narrow corridor by patching the collision checker.
    # The "corridor" is at y=40.73. Any deviation from this latitude is a "collision".
    original_is_line_obstructed = real_environment.is_line_obstructed

    def corridor_collision_check(p1, p2):
        # If the original check finds a collision, respect it.
        if original_is_line_obstructed(p1, p2):
            return True
        # Add our virtual wall constraint: path must stay on the y=40.73 line.
        if abs(p1[1] - 40.73) > 0.0001 or abs(p2[1] - 40.73) > 0.0001:
            return True # It's a collision with the virtual wall
        return False

    agent1 = Agent(id="DroneA", start_pos=(-74.01, 40.73, 50), goal_pos=(-73.99, 40.73, 50), config={})
    agent2 = Agent(id="DroneB", start_pos=(-73.99, 40.73, 50), goal_pos=(-74.01, 40.73, 50), config={})
    
    planner = CBSHPlanner(real_environment, real_coord_manager)

    with patch.object(real_environment, 'is_line_obstructed', side_effect=corridor_collision_check):
        # ACT: Attempt to plan the unsolvable mission
        solution = planner.plan_fleet([agent1, agent2])

    # ASSERT: The planner should exhaust its search and return None.
    assert solution is None

def test_cbsh_handles_single_agent_case(real_cbsh_planner):
    """
    Boundary Value: The multi-agent planner should function correctly as a
    single-agent planner when only one agent is provided.
    """
    agent = Agent(id="SoloDrone", start_pos=(-74.01, 40.73, 50), goal_pos=(-73.99, 40.73, 50), config={})
    
    solution = real_cbsh_planner.plan_fleet([agent])

    assert solution is not None
    assert "SoloDrone" in solution
    assert len(solution["SoloDrone"]) > 1 # Path should have at least start and end

def test_fleet_manager_handles_partial_failure(real_environment, real_coord_manager):
    """
    Edge Case: If one mission in a fleet is impossible, the entire planning
    cycle should fail, and the correct mission state should be set.
    """
    # ARRANGE
    mock_planner = CBSHPlanner(real_environment, real_coord_manager)
    predictor = EnergyTimePredictor()
    fm = FleetManager(mock_planner, predictor)

    # Mission 1 is possible, Mission 2 is impossible
    mission_ok = Mission("DroneOK", (-74.0, 40.73, 50), [(-73.99, 40.73, 50)], 1.0, {})
    mission_fail = Mission("DroneFail", (-74.0, 40.72, 50), [(-74.00, 40.72, 50.0)], 1.0, {}) # Goal is in NFZ
    fm.add_mission(mission_ok)
    fm.add_mission(mission_fail)
    
    # We patch the planner to return None, simulating a top-level failure
    with patch.object(mock_planner, 'plan_fleet', return_value=None) as mock_plan_method:
        # ACT
        success = fm.execute_planning_cycle()

        # ASSERT
        assert success is False
        mock_plan_method.assert_called_once()
        
        # The manager should have identified the agents that were part of the failed plan
        # and marked their missions accordingly.
        assert fm.missions["DroneOK"].state == "PLANNING_FAILED"
        assert fm.missions["DroneFail"].state == "PLANNING_FAILED"