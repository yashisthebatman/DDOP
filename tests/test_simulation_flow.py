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

# A stand-in for the real app's update function to test the logic
from app import update_simulation


@pytest.fixture
def test_dependencies():
    """Provides a complete set of integrated components for testing."""
    coord_manager = CoordinateManager()
    weather = WeatherSystem(seed=123)
    env = Environment(weather, coord_manager)
    predictor = EnergyTimePredictor()
    # Mock the planner for speed and predictability in tests
    mock_planner = MagicMock(spec=CBSHPlanner)
    
    mock_planner.smoother = MagicMock()
    mock_planner.smoother.smooth_path.side_effect = lambda path, env: path
    
    mock_planner.env = env

    fleet_manager = FleetManager(mock_planner, predictor)
    # Use the REAL VRP solver for a true integration test
    vrp_solver = VRPSolver(predictor)
    dispatcher = Dispatcher(vrp_solver)
    return {
        "fleet_manager": fleet_manager,
        "dispatcher": dispatcher,
        "mock_planner": mock_planner
    }


def test_full_mission_lifecycle(test_dependencies):
    """
    Tests the entire flow: dispatch -> plan -> simulate -> complete.
    """
    state = get_initial_state()
    fm = test_dependencies['fleet_manager']
    dispatcher = test_dependencies['dispatcher']
    mock_planner = test_dependencies['mock_planner']

    # Ensure at least one drone is IDLE and ready.
    # The VRP solver requires drones to start at a valid hub location.
    state['drones']['Drone 1']['status'] = 'IDLE'
    
    # State Setup: Create enough orders to trigger the dispatcher
    state['pending_orders'] = {
        "Order1": {'id': "Order1", 'pos': DESTINATIONS['NYU Campus'], 'payload_kg': 1.0},
        "Order2": {'id': "Order2", 'pos': DESTINATIONS['Union Square'], 'payload_kg': 1.0},
        "Order3": {'id': "Order3", 'pos': DESTINATIONS['Chelsea Market'], 'payload_kg': 1.0},
        "Order4": {'id': "Order4", 'pos': DESTINATIONS['Wall Street Bull'], 'payload_kg': 1.0},
        "Order5": {'id': "Order5", 'pos': DESTINATIONS['South Street Seaport'], 'payload_kg': 1.0}
    }
    state['simulation_time'] = 0.0

    # 1. Dispatch the order
    dispatched = dispatcher.dispatch_missions(state)
    assert dispatched is True
    
    # FIX: Don't assume which drone gets the mission. Find the one that is PLANNING.
    assigned_drone = next((d for d in state['drones'].values() if d['status'] == 'PLANNING'), None)
    assert assigned_drone is not None
    assigned_drone_id = assigned_drone['id']
    
    assert "Order1" in state['pending_orders'] # Order is NOT removed yet

    # 2. Plan the mission (simulating the async call)
    mission_id = assigned_drone['mission_id']
    final_dest = state['active_missions'][mission_id]['destinations'][-1]
    mock_path = [assigned_drone['pos'], final_dest]
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
    mission_order_ids = state['active_missions'][mission_id]['order_ids']
    assert not any(oid in state['pending_orders'] for oid in mission_order_ids)

    # 3. Simulate drone movement
    initial_pos = tuple(state['drones'][assigned_drone_id]['pos'])
    update_simulation(state, fm)
    assert tuple(state['drones'][assigned_drone_id]['pos']) != initial_pos

    # 4. Simulate mission completion
    state['simulation_time'] = 100 
    update_simulation(state, fm)
    
    assert state['drones'][assigned_drone_id]['status'] == 'RECHARGING'
    assert not state['active_missions']
    assert mission_id in state['completed_missions']
    for order_id in state['completed_missions'][mission_id]['order_ids']:
        assert order_id in state['completed_orders']


def test_multi_stop_mission_completion(test_dependencies):
    """
    Tests that for a multi-stop mission, all associated orders are marked
    as complete when the drone finishes its single, consolidated path.
    """
    state = get_initial_state()
    fm = test_dependencies['fleet_manager']
    dispatcher = test_dependencies['dispatcher']
    mock_planner = test_dependencies['mock_planner']
    
    # Ensure all drones are initially idle to give the solver options.
    for drone in state['drones'].values():
        drone['status'] = "IDLE"

    state['pending_orders'] = {
        "OrderA": {'id': "OrderA", 'pos': DESTINATIONS['Wall Street Bull'], 'payload_kg': 1.0},
        "OrderB": {'id': "OrderB", 'pos': DESTINATIONS['South Street Seaport'], 'payload_kg': 1.0},
        "OrderC": {'id': "OrderC", 'pos': DESTINATIONS['NYU Campus'], 'payload_kg': 1.0},
        "OrderD": {'id': "OrderD", 'pos': DESTINATIONS['Union Square'], 'payload_kg': 1.0},
        "OrderE": {'id': "OrderE", 'pos': DESTINATIONS['Chelsea Market'], 'payload_kg': 1.0}
    }

    dispatcher.dispatch_missions(state)

    # FIX: Find the drone that was actually assigned the mission instead of assuming it's "Drone 1".
    assigned_drone = next((d for d in state['drones'].values() if d['status'] == 'PLANNING'), None)
    assert assigned_drone is not None, "Dispatcher failed to assign any drone to PLANNING state."
    assigned_drone_id = assigned_drone['id']
    
    mission_id = assigned_drone['mission_id']
    mission = state['active_missions'][mission_id]
    assert len(mission['order_ids']) >= 1 

    final_dest = mission['destinations'][-1]
    mock_path = [assigned_drone['pos'], final_dest]
    mock_solution = {assigned_drone_id: [(p, i * 100) for i, p in enumerate(mock_path)]}
    mock_planner.plan_fleet.return_value = mock_solution
    success, plan_results = fm.plan_pending_missions(state)
    
    state['drones'][assigned_drone_id].update(plan_results['drone_updates'][assigned_drone_id])
    state['active_missions'][mission_id].update(plan_results['mission_updates'][mission_id])

    state['simulation_time'] = 200
    update_simulation(state, fm)

    assert mission_id in state['completed_missions']
    completed_mission = state['completed_missions'][mission_id]
    for order_id in completed_mission['order_ids']:
        assert order_id in state['completed_orders']