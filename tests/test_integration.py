# FILE: tests/test_integration.py
import pytest
import numpy as np
import time
from unittest.mock import MagicMock
import threading 

from system_state import get_initial_state
from environment import Environment, WeatherSystem
from utils.coordinate_manager import CoordinateManager
from ml_predictor.predictor import EnergyTimePredictor
from planners.cbsh_planner import CBSHPlanner, MIN_SEPARATION_METERS
# FIX: Import Mission to create a real object
from fleet.manager import FleetManager, Mission
from fleet.cbs_components import Agent
from config import NO_FLY_ZONES, HUBS, DESTINATIONS
from server import update_simulation
import simulation.contingency_planner as contingency_planner


@pytest.fixture(scope="module")
def real_coord_manager():
    return CoordinateManager()

@pytest.fixture(scope="module")
def real_environment(real_coord_manager):
    env = Environment(WeatherSystem(seed=42), real_coord_manager)
    env.buildings = []
    env.obstacles = {}
    from rtree import index
    p = index.Property()
    p.dimension = 3
    env.obstacle_index = index.Index(properties=p)
    env.obstacle_counter = 0 
    env._index_static_nfzs()
    return env

@pytest.fixture
def real_cbsh_planner(real_environment, real_coord_manager):
    return CBSHPlanner(real_environment, real_coord_manager)

@pytest.fixture
def real_fleet_manager(real_cbsh_planner):
    predictor = EnergyTimePredictor()
    predictor.load_model()
    mock_lock = threading.Lock()
    return FleetManager(real_cbsh_planner, predictor, mock_lock)


@pytest.mark.slow
def test_full_stack_solves_head_on_conflict_with_real_planner(real_cbsh_planner, real_coord_manager):
    agent1 = Agent(id="DroneA", start_pos=HUBS["Hub A (South Manhattan)"], goal_pos=DESTINATIONS["NYU Campus"], config={})
    agent2 = Agent(id="DroneB", start_pos=HUBS["Hub C (West Side)"], goal_pos=DESTINATIONS["South Street Seaport"], config={})
    solution = real_cbsh_planner.plan_fleet([agent1, agent2])
    assert solution is not None
    assert "DroneA" in solution and solution["DroneA"] is not None
    assert "DroneB" in solution and solution["DroneB"] is not None
    path_a = real_cbsh_planner._get_interpolated_path(solution["DroneA"])
    path_b = real_cbsh_planner._get_interpolated_path(solution["DroneB"])
    max_len = min(len(path_a), len(path_b))
    for t in range(max_len):
        if path_a[t] is not None and path_b[t] is not None:
            pos_a_m = real_coord_manager.world_to_meters(path_a[t])
            pos_b_m = real_coord_manager.world_to_meters(path_b[t])
            distance = np.linalg.norm(np.array(pos_a_m) - np.array(pos_b_m))
            assert distance > MIN_SEPARATION_METERS

def test_system_handles_unplannable_mission_gracefully(real_fleet_manager):
    state = get_initial_state()
    nfz = NO_FLY_ZONES[0]
    obstructed_goal = ( (nfz[0] + nfz[2]) / 2, (nfz[1] + nfz[3]) / 2, 50.0 )
    
    # FIX: Use a real Mission object, not a mock
    mission_obj = Mission(
        mission_id='M-IMPOSSIBLE', drone_id='Drone 1',
        start_pos=HUBS["Hub A (South Manhattan)"],
        destinations=[obstructed_goal], payload_kg=1.0, order_ids=[]
    )
    
    real_fleet_manager.add_mission_to_queue(mission_obj)
    success, results = real_fleet_manager.plan_pending_missions(state)
    
    assert success is False
    assert results['drone_updates']['Drone 1']['status'] == 'IDLE'
    assert 'M-IMPOSSIBLE' in results['mission_failures']

def test_system_reacts_to_dynamic_nfz_mid_mission(real_environment, real_fleet_manager):
    state = get_initial_state()
    drone_id = "Drone 1"
    
    start_pos = HUBS["Hub A (South Manhattan)"]
    goal_pos = DESTINATIONS["StuyTown Apartments"]
    
    # FIX: Use a real Mission object
    mission_obj = Mission(
        mission_id="M-DYNAMIC", drone_id=drone_id, start_pos=start_pos,
        destinations=[goal_pos], payload_kg=1.0, order_ids=['OrderX']
    )
    mission_obj.stops = [{'id': 'OrderX', 'pos': goal_pos}]

    real_fleet_manager.add_mission_to_queue(mission_obj)
    success, plan_results = real_fleet_manager.plan_pending_missions(state)
    assert success is True

    mission_id = mission_obj.mission_id
    state['drones'][drone_id].update(plan_results['drone_updates'][drone_id])
    state['active_missions'][mission_id].update(plan_results['mission_updates'][mission_id])
    assert state['drones'][drone_id]['status'] == 'EN ROUTE'

    path = state['active_missions'][mission_id]['path_world_coords']
    midpoint_idx = len(path) // 2
    midpoint = path[midpoint_idx]
    
    nfz_bounds = [midpoint[0] - 0.001, midpoint[1] - 0.001, midpoint[0] + 0.001, midpoint[1] + 0.001]
    real_environment.add_dynamic_nfz(nfz_bounds)

    planners_dict = {
        "env": real_environment,
        "predictor": real_fleet_manager.predictor,
        "coord_manager": real_environment.coord_manager
    }
    contingency_planner.check_for_contingencies(state, planners_dict)

    assert state['drones'][drone_id]['status'] == 'EMERGENCY_RETURN'
    assert mission_id not in state['active_missions']
    assert 'OrderX' in state['pending_orders']