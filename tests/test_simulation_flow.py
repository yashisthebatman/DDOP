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
    
    # FIX: Explicitly create the 'env' attribute on the mock planner.
    # The FleetManager needs this to pass to the path smoother.
    mock_planner.env = env

    fleet_manager = FleetManager(mock_planner, predictor)
    vrp_solver = VRPSolver(predictor)
    dispatcher = Dispatcher(vrp_solver)
    return {
        "fleet_manager": fleet_manager,
        "dispatcher": dispatcher,
        "mock_planner": mock_planner
    }


def test_full_mission_lifecycle(test_dependencies):
    """
    Tests the entire flow:
    1. An order is created.
    2. The dispatcher assigns it to a drone (status -> PLANNING).
    3. The fleet manager plans a path (status -> EN ROUTE).
    4. The simulation updates the drone's position.
    5. The mission completes, and the drone becomes available again.
    """
    state = get_initial_state()
    fm = test_dependencies['fleet_manager']
    dispatcher = test_dependencies['dispatcher']
    mock_planner = test_dependencies['mock_planner']

    # State Setup: One idle drone, one order
    state['drones'] = {
        "D1": {'id': "D1", 'pos': HUBS['Hub A (South Manhattan)'], 'home_hub': 'Hub A (South Manhattan)', 'battery': DRONE_BATTERY_WH, 'max_payload_kg': DRONE_MAX_PAYLOAD_KG, 'status': 'IDLE', 'mission_id': None, 'available_at': 0.0}
    }
    state['pending_orders'] = {
        "Order1": {'id': "Order1", 'pos': DESTINATIONS['NYU Campus'], 'payload_kg': 2.0}
    }
    state['simulation_time'] = 0.0

    # 1. Dispatch the order
    # To meet the minimum batch size for the dispatcher
    state['pending_orders']["Order2"] = {'id': "Order2", 'pos': DESTINATIONS['Union Square'], 'payload_kg': 1.0}
    state['pending_orders']["Order3"] = {'id': "Order3", 'pos': DESTINATIONS['Chelsea Market'], 'payload_kg': 1.0}
    dispatched = dispatcher.dispatch_missions(state)
    assert dispatched is True
    assert state['drones']['D1']['status'] == 'PLANNING'
    assert "Order1" in state['pending_orders'] # Order is NOT removed yet

    # 2. Plan the mission (simulating the async call)
    mission_id = state['drones']['D1']['mission_id']
    # The VRP solver might group multiple orders, so we plan to the final destination in the list
    final_dest = state['active_missions'][mission_id]['destinations'][-1]
    mock_path = [state['drones']['D1']['pos'], final_dest]
    mock_solution = {"D1": [(p, i * 60) for i, p in enumerate(mock_path)]} # Path takes 60s
    mock_planner.plan_fleet.return_value = mock_solution
    
    # Run the planner logic
    success, plan_results = fm.plan_pending_missions(state)
    assert success is True
    
    # Apply the plan results to the state (mimicking the main app loop)
    drone_updates = plan_results['drone_updates']
    mission_updates = plan_results['mission_updates']
    state['drones']['D1'].update(drone_updates['D1'])
    state['active_missions'][mission_id].update(mission_updates[mission_id])
    for mid in plan_results.get('successful_mission_ids', []):
        order_ids = state['active_missions'][mid]['order_ids']
        for oid in order_ids:
            if oid in state['pending_orders']:
                del state['pending_orders'][oid]

    assert state['drones']['D1']['status'] == 'EN ROUTE'
    # Assert that all orders dispatched in the mission are removed from pending
    mission_order_ids = state['active_missions'][mission_id]['order_ids']
    assert not any(oid in state['pending_orders'] for oid in mission_order_ids)

    # 3. Simulate drone movement
    initial_pos = tuple(state['drones']['D1']['pos'])
    update_simulation(state, fm) # Update at t=0.5
    assert tuple(state['drones']['D1']['pos']) != initial_pos # Position should change

    # 4. Simulate mission completion
    state['simulation_time'] = 100 # Fast-forward past the 60s mission time
    update_simulation(state, fm)
    
    assert state['drones']['D1']['status'] == 'RECHARGING'
    assert not state['active_missions']
    assert mission_id in state['completed_missions']
    # Assert all orders from the completed mission are in the completed list
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
    
    # State Setup: One drone, two orders close together
    state['drones'] = { "D1": {'id': "D1", 'pos': HUBS['Hub A (South Manhattan)'], 'home_hub': 'Hub A (South Manhattan)', 'battery': DRONE_BATTERY_WH, 'max_payload_kg': DRONE_MAX_PAYLOAD_KG, 'status': 'IDLE', 'mission_id': None, 'available_at': 0.0}}
    state['pending_orders'] = {
        "OrderA": {'id': "OrderA", 'pos': DESTINATIONS['Wall Street Bull'], 'payload_kg': 1.0},
        "OrderB": {'id': "OrderB", 'pos': DESTINATIONS['South Street Seaport'], 'payload_kg': 1.0},
        "OrderC": {'id': "OrderC", 'pos': DESTINATIONS['NYU Campus'], 'payload_kg': 1.0}
    }
    # Dispatcher creates a single mission for both OrderA and OrderB
    dispatcher.dispatch_missions(state)
    assert state['drones']['D1']['status'] == 'PLANNING'
    mission_id = state['drones']['D1']['mission_id']
    mission = state['active_missions'][mission_id]
    assert len(mission['order_ids']) >= 2

    # Plan the mission
    final_dest = mission['destinations'][-1]
    mock_path = [state['drones']['D1']['pos'], final_dest]
    mock_solution = {"D1": [(p, i * 100) for i, p in enumerate(mock_path)]}
    mock_planner.plan_fleet.return_value = mock_solution
    success, plan_results = fm.plan_pending_missions(state)
    
    # Apply results
    state['drones']['D1'].update(plan_results['drone_updates']['D1'])
    state['active_missions'][mission_id].update(plan_results['mission_updates'][mission_id])

    # Simulate completion
    state['simulation_time'] = 200
    update_simulation(state, fm)

    # Assert both orders from the mission are completed
    assert mission_id in state['completed_missions']
    completed_mission = state['completed_missions'][mission_id]
    for order_id in completed_mission['order_ids']:
        assert order_id in state['completed_orders']