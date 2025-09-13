# FILE: tests/test_integration.py
import pytest
import numpy as np
import time
from unittest.mock import MagicMock

# --- Import all the real components we need for a full-stack test ---
from system_state import get_initial_state
from environment import Environment, WeatherSystem
from utils.coordinate_manager import CoordinateManager
from ml_predictor.predictor import EnergyTimePredictor
from planners.cbsh_planner import CBSHPlanner, MIN_SEPARATION_METERS
from fleet.manager import FleetManager
from fleet.cbs_components import Agent
from config import NO_FLY_ZONES, HUBS, DESTINATIONS
from server import update_simulation # <--- FIX: Changed 'app' to 'server'
import simulation.contingency_planner as contingency_planner


# --- Fixtures to provide REAL, SHARED instances of our components ---

@pytest.fixture(scope="module")
def real_coord_manager():
    """Provides a single, real CoordinateManager instance for the entire test module."""
    return CoordinateManager()

@pytest.fixture(scope="module")
def real_environment(real_coord_manager):
    """
    Provides a real Environment but with NO random buildings to ensure predictable
    test paths.
    """
    env = Environment(WeatherSystem(seed=42), real_coord_manager)
    # Clear buildings to ensure straight paths for conflict/NFZ tests.
    env.buildings = []
    env.obstacles = {}
    
    # Reinitialize the spatial index without buildings
    from rtree import index
    p = index.Property()
    p.dimension = 3
    env.obstacle_index = index.Index(properties=p)
    env.obstacle_counter = 0 # Reset counter
    env._index_static_nfzs()  # Only static NFZs are indexed
    
    return env

@pytest.fixture
def real_cbsh_planner(real_environment, real_coord_manager):
    """Provides a fresh, real CBSHPlanner for each test function."""
    return CBSHPlanner(real_environment, real_coord_manager)

@pytest.fixture
def real_fleet_manager(real_cbsh_planner):
    """Provides a real FleetManager using the real planner."""
    predictor = EnergyTimePredictor()
    predictor.load_model() # Load the actual model
    return FleetManager(real_cbsh_planner, predictor)


# --- Full-Stack Integration Tests ---

@pytest.mark.slow
def test_full_stack_solves_head_on_conflict_with_real_planner(real_cbsh_planner, real_coord_manager):
    """
    A deep integration test to verify the entire planning stack (CBSH -> A* -> RRT* -> TimingSolver)
    can deconflict two drones in a real environment with obstacles.
    """
    # SCENARIO: Two drones are given paths that will cross.
    agent1 = Agent(id="DroneA", start_pos=HUBS["Hub A (South Manhattan)"], goal_pos=DESTINATIONS["NYU Campus"], config={})
    agent2 = Agent(id="DroneB", start_pos=HUBS["Hub C (West Side)"], goal_pos=DESTINATIONS["South Street Seaport"], config={})
    
    # ACTION: Run the full planning process.
    solution = real_cbsh_planner.plan_fleet([agent1, agent2])

    # ASSERTION: A valid, deconflicted solution must be found.
    assert solution is not None, "Real planner should have found a solution for the head-on conflict."
    assert "DroneA" in solution and solution["DroneA"] is not None
    assert "DroneB" in solution and solution["DroneB"] is not None

    # Verify that the drones never violate the safety bubble.
    path_a = real_cbsh_planner._get_interpolated_path(solution["DroneA"])
    path_b = real_cbsh_planner._get_interpolated_path(solution["DroneB"])
    max_len = min(len(path_a), len(path_b))

    for t in range(max_len):
        pos_a_world = path_a[t]
        pos_b_world = path_b[t]
        if pos_a_world is not None and pos_b_world is not None:
            pos_a_m = real_coord_manager.world_to_meters(pos_a_world)
            pos_b_m = real_coord_manager.world_to_meters(pos_b_world)
            distance = np.linalg.norm(np.array(pos_a_m) - np.array(pos_b_m))
            assert distance > MIN_SEPARATION_METERS, f"Conflict at time {t}! Drones are {distance:.2f}m apart."

def test_system_handles_unplannable_mission_gracefully(real_fleet_manager):
    """
    Tests the boundary condition where a mission is impossible to plan
    (e.g., goal is inside a static obstacle). The system should fail gracefully.
    """
    state = get_initial_state()
    # SCENARIO: Create a mission where the destination is inside a known No-Fly Zone.
    nfz = NO_FLY_ZONES[0]
    obstructed_goal = ( (nfz[0] + nfz[2]) / 2, (nfz[1] + nfz[3]) / 2, 50.0 )
    
    state['drones']['Drone 1']['status'] = 'PLANNING'
    state['drones']['Drone 1']['mission_id'] = 'M-IMPOSSIBLE'
    state['active_missions']['M-IMPOSSIBLE'] = {
        'drone_id': 'Drone 1', 'start_pos': HUBS["Hub A (South Manhattan)"],
        'destinations': [obstructed_goal], 'payload_kg': 1.0, 'stops': [], 'order_ids': []
    }

    # ACTION: Attempt to plan the impossible mission.
    success, results = real_fleet_manager.plan_pending_missions(state)
    
    # ASSERTION: The planning should fail, and the drone should be returned to an IDLE state.
    assert success is False, "Planning should fail for a goal inside an obstacle."
    assert results['drone_updates']['Drone 1']['status'] == 'IDLE'
    assert 'M-IMPOSSIBLE' in results['mission_failures']

def test_system_reacts_to_dynamic_nfz_mid_mission(real_environment, real_fleet_manager):
    """
    Tests the integration of the simulation loop with the contingency planner.
    A drone's mission should be cancelled if a new NFZ blocks its path.
    """
    state = get_initial_state()
    drone_id = "Drone 1"
    mission_id = "M-DYNAMIC"
    
    # SCENARIO 1: Plan a valid mission.
    start_pos = HUBS["Hub A (South Manhattan)"]
    goal_pos = DESTINATIONS["StuyTown Apartments"] # A clear path exists for this.
    state['drones'][drone_id]['status'] = 'PLANNING'
    state['drones'][drone_id]['mission_id'] = mission_id
    state['active_missions'][mission_id] = {
        'drone_id': drone_id, 'start_pos': start_pos, 'destinations': [goal_pos], 
        'payload_kg': 1.0, 'stops': [{'id': 'OrderX', 'pos': goal_pos}], 'order_ids': ['OrderX']
    }
    
    success, plan_results = real_fleet_manager.plan_pending_missions(state)
    assert success is True, "Initial planning for the mission should succeed."

    # Update state to reflect a successful plan
    state['drones'][drone_id].update(plan_results['drone_updates'][drone_id])
    state['active_missions'][mission_id].update(plan_results['mission_updates'][mission_id])
    assert state['drones'][drone_id]['status'] == 'EN ROUTE'

    # SCENARIO 2: Advance simulation and inject a dynamic obstacle.
    # Get the drone's path and find a midpoint to block.
    path = state['active_missions'][mission_id]['path_world_coords']
    midpoint_idx = len(path) // 2
    midpoint = path[midpoint_idx]
    
    # Create an NFZ centered on the drone's future path.
    nfz_bounds = [midpoint[0] - 0.001, midpoint[1] - 0.001, midpoint[0] + 0.001, midpoint[1] + 0.001]
    real_environment.add_dynamic_nfz(nfz_bounds)

    # ACTION: Run the contingency checker.
    # The 'planners' dict needs to be assembled for the function call.
    planners_dict = {
        "env": real_environment,
        "predictor": real_fleet_manager.predictor,
        "coord_manager": real_environment.coord_manager
    }
    contingency_planner.check_for_contingencies(state, planners_dict)

    # ASSERTION: The drone should have entered an emergency state, and the mission cancelled.
    assert state['drones'][drone_id]['status'] == 'EMERGENCY_RETURN'
    assert mission_id not in state['active_missions']
    assert 'OrderX' in state['pending_orders'], "The order from the failed mission should be re-queued."