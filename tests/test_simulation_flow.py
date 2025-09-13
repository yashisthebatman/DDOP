# FILE: tests/test_simulation_flow.py
import pytest
import time
from unittest.mock import MagicMock, patch

# --- Import all the real components we need to integrate ---
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

    fleet_manager = FleetManager(mock_planner, predictor)
    vrp_solver = VRPSolver(predictor)
    dispatcher = Dispatcher(vrp_solver)
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

    state['drones']['Drone 1']['status'] = 'IDLE'
    
    # FIX: Added 'dest_name' to each order to make the mock data complete.
    state['pending_orders'] = {
        "Order1": {'id': "Order1", 'pos': DESTINATIONS['NYU Campus'], 'payload_kg': 1.0, 'dest_name': 'NYU Campus'},
        "Order2": {'id': "Order2", 'pos': DESTINATIONS['Union Square'], 'payload_kg': 1.0, 'dest_name': 'Union Square'},
        "Order3": {'id': "Order3", 'pos': DESTINATIONS['Chelsea Market'], 'payload_kg': 1.0, 'dest_name': 'Chelsea Market'},
        "Order4": {'id': "Order4", 'pos': DESTINATIONS['Wall Street Bull'], 'payload_kg': 1.0, 'dest_name': 'Wall Street Bull'},
        "Order5": {'id': "Order5", 'pos': DESTINATIONS['South Street Seaport'], 'payload_kg': 1.0, 'dest_name': 'South Street Seaport'}
    }
    state['simulation_time'] = 0.0

    dispatched = dispatcher.dispatch_missions(state)
    assert dispatched is True

    assigned_drone = next((d for d in state['drones'].values() if d['status'] == 'PLANNING'), None)
    assert assigned_drone is not None
    
    assigned_drone_id = assigned_drone['id']
    mission_id = assigned_drone['mission_id']
    mission = state['active_missions'][mission_id]
    
    mock_path = [assigned_drone['pos']]
    mock_path.extend([stop['pos'] for stop in mission['stops']])
    mock_path.append(mission['destinations'][-1])

    mock_solution = {assigned_drone_id: [(p, i * 60) for i, p in enumerate(mock_path)]}
    mock_planner.plan_fleet.return_value = mock_solution
    
    success, plan_results = fm.plan_pending_missions(state)
    assert success is True
    
    drone_updates = plan_results['drone_updates']
    mission_updates = plan_results['mission_updates']
    state['drones'][assigned_drone_id].update(drone_updates[assigned_drone_id])
    state['active_missions'][mission_id].update(mission_updates[mission_id])
    for mid in plan_results.get('successful_mission_ids', []):
        order_ids = state['active_missions'][mid]['order_ids']
        for oid in order_ids:
            if oid in state['pending_orders']:
                del state['pending_orders'][oid]

    assert state['drones'][assigned_drone_id]['status'] == 'EN ROUTE'
    
    max_loops = 5000 # Increased loop count to be safe
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