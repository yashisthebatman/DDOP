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
    return CoordinateManager()

@pytest.fixture(scope="module")
def real_environment(real_coord_manager):
    from environment import WeatherSystem
    return Environment(WeatherSystem(seed=123), real_coord_manager)

@pytest.fixture
def real_cbsh_planner(real_environment, real_coord_manager):
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

    path_a = real_cbsh_planner._get_interpolated_path(solution["DroneA"])
    path_b = real_cbsh_planner._get_interpolated_path(solution["DroneB"])
    max_len = min(len(path_a), len(path_b))

    for t in range(max_len):
        pos_a = path_a[t]
        pos_b = path_b[t]
        if pos_a is not None and pos_b is not None:
            # FIX: Use the new CoordinateManager API: world_to_meters
            pos_a_m = real_coord_manager.world_to_meters(pos_a)
            pos_b_m = real_coord_manager.world_to_meters(pos_b)
            distance = np.linalg.norm(np.array(pos_a_m) - np.array(pos_b_m))
            assert distance > MIN_SEPARATION_METERS, f"Conflict at time {t}! Drones are {distance:.2f}m apart."


def test_fleet_manager_updates_missions_on_success(real_environment, real_coord_manager):
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
    obstructed_goal = (-74.00, 40.72, 50.0)
    agent = Agent(id="DroneC", start_pos=(-74.02, 40.72, 50), goal_pos=obstructed_goal, config={})
    solution = real_cbsh_planner.plan_fleet([agent])
    assert solution is None

@pytest.mark.slow
def test_cbsh_terminates_for_unsolvable_conflict(real_environment, real_coord_manager):
    """
    FIX: This test is modified to be logically correct. It now tests that CBS
    terminates when a conflict is both geometrically AND temporally unsolvable.
    """
    agent1 = Agent(id="DroneA", start_pos=(-74.01, 40.73, 50), goal_pos=(-73.99, 40.73, 50), config={})
    agent2 = Agent(id="DroneB", start_pos=(-73.99, 40.73, 50), goal_pos=(-74.01, 40.73, 50), config={})
    
    planner = CBSHPlanner(real_environment, real_coord_manager)
    original_find_path = planner._find_path_for_agent

    # Mock the low-level planner. It will succeed for DroneA but then we make it
    # fail for DroneB, simulating an unsolvable temporal problem.
    def mock_find_path(agent, constraints):
        if agent.id == "DroneB":
            return None # Simulate failure for the second agent's replan
        return original_find_path(agent, constraints)

    with patch.object(planner, '_find_path_for_agent', side_effect=mock_find_path):
        solution = planner.plan_fleet([agent1, agent2])

    # ASSERT: The planner should exhaust its search and return None.
    assert solution is None

def test_cbsh_handles_single_agent_case(real_cbsh_planner):
    agent = Agent(id="SoloDrone", start_pos=(-74.01, 40.73, 50), goal_pos=(-73.99, 40.73, 50), config={})
    solution = real_cbsh_planner.plan_fleet([agent])
    assert solution is not None
    assert "SoloDrone" in solution
    assert len(solution["SoloDrone"]) > 1

def test_fleet_manager_handles_partial_failure(real_environment, real_coord_manager):
    mock_planner = CBSHPlanner(real_environment, real_coord_manager)
    predictor = EnergyTimePredictor()
    fm = FleetManager(mock_planner, predictor)
    mission_ok = Mission("DroneOK", (-74.0, 40.73, 50), [(-73.99, 40.73, 50)], 1.0, {})
    mission_fail = Mission("DroneFail", (-74.0, 40.72, 50), [(-74.00, 40.72, 50.0)], 1.0, {}) # Goal in NFZ
    fm.add_mission(mission_ok)
    fm.add_mission(mission_fail)
    
    with patch.object(mock_planner, 'plan_fleet', return_value=None) as mock_plan_method:
        success = fm.execute_planning_cycle()
        assert success is False
        mock_plan_method.assert_called_once()
        assert fm.missions["DroneOK"].state == "PLANNING_FAILED"
        assert fm.missions["DroneFail"].state == "PLANNING_FAILED"