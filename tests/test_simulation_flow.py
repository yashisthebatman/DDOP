# FILE: tests/test_simulation_flow.py
import pytest
import time
import threading 
from unittest.mock import MagicMock, patch

from config import DRONE_BATTERY_WH, DRONE_MAX_PAYLOAD_KG, HUBS, DESTINATIONS
from system_state import get_initial_state
from dispatch.dispatcher import Dispatcher
from dispatch.vrp_solver import VRPSolver
from fleet.manager import FleetManager
from planners.cbsh_planner import CBSHPlanner
from ml_predictor.predictor import EnergyTimePredictor
from environment import Environment, WeatherSystem
from utils.coordinate_manager import CoordinateManager
from server import update_simulation


@pytest.fixture
def test_dependencies():
    """Provides a complete set of integrated components for testing."""
    coord_manager = CoordinateManager()
    weather = WeatherSystem(seed=123)
    env = Environment(weather, coord_manager)
    predictor = EnergyTimePredictor()
    predictor.load_model()
    mock_planner = MagicMock(spec=CBSHPlanner)
    
    mock_planner.smoother = MagicMock()
    mock_planner.smoother.smooth_path.side_effect = lambda path, env: path
    mock_planner.env = env

    mock_lock = threading.Lock()
    fleet_manager = FleetManager(mock_planner, predictor, mock_lock)
    vrp_solver = VRPSolver(predictor)
    dispatcher = Dispatcher(vrp_solver, fleet_manager)
    
    return {
        "fleet_manager": fleet_manager,
        "dispatcher": dispatcher,
        "mock_planner": mock_planner,
        "coord_manager": coord_manager,
        "env": env,
        "predictor": predictor
    }


def test_full_mission_lifecycle(test_dependencies):
    """
    Tests the entire flow: dispatch -> plan -> simulate -> complete.
    """
    state = get_initial_state()
    fm = test_dependencies['fleet_manager']
    dispatcher = test_dependencies['dispatcher']
    mock_planner = test_dependencies['mock_planner']
    
    for drone in state['drones'].values():
        drone['status'] = 'IDLE'

    state['pending_orders'] = {
        "Order1": {'id': "Order1", 'pos': DESTINATIONS['NYU Campus'], 'payload_kg': 1.0, 'dest_name': 'NYU Campus'},
        "Order2": {'id': "Order2", 'pos': DESTINATIONS['Union Square'], 'payload_kg': 1.0, 'dest_name': 'Union Square'},
    }
    state['simulation_time'] = 0.0

    dispatched = dispatcher.dispatch_missions(state)
    assert dispatched is True
    assert not fm.planning_queue.empty()

    _, mission_obj = fm.planning_queue.queue[0]
    assigned_drone_id = mission_obj.drone_id

    # --- THIS IS THE FIX ---
    # The mock path must be consistent with the multi-stop mission from the VRP solver.
    drone_pos = state['drones'][assigned_drone_id]['pos']
    # Create a list of all stop positions from the mission object
    stop_positions = [stop['pos'] for stop in mission_obj.stops]
    end_hub_pos = HUBS[mission_obj.end_hub]
    # The full path includes the start, all stops, and the final hub
    mock_path = [drone_pos] + stop_positions + [end_hub_pos]
    
    mock_solution = {assigned_drone_id: [(p, i * 60) for i, p in enumerate(mock_path)]}
    mock_planner.plan_fleet.return_value = mock_solution

    success, plan_results = fm.plan_pending_missions(state)
    assert success is True

    mission_id = plan_results['successful_mission_ids'][0]

    state['drones'][assigned_drone_id].update(plan_results['drone_updates'][assigned_drone_id])
    state['active_missions'][mission_id].update(plan_results['mission_updates'][mission_id])
    for oid in state['active_missions'][mission_id]['order_ids']:
        if oid in state['pending_orders']: del state['pending_orders'][oid]

    assert state['drones'][assigned_drone_id]['status'] == 'EN ROUTE'

    max_loops = 5000
    loop_count = 0
    while mission_id in state['active_missions'] and loop_count < max_loops:
        update_simulation(state, test_dependencies)
        loop_count += 1

    assert loop_count < max_loops, "Simulation timed out; mission never completed."
    
    assert state['drones'][assigned_drone_id]['status'] == 'RECHARGING'
    assert mission_id not in state['active_missions']
    assert mission_id in state['completed_missions']
    for order_id in state['completed_missions'][mission_id]['order_ids']:
        assert order_id in state['completed_orders']