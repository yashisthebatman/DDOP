# FILE: tests/test_simulation_flow.py
import pytest
import time
import threading
from unittest.mock import MagicMock, patch

from config import HUBS, DESTINATIONS
from system_state import get_initial_state
from dispatch.dispatcher import Dispatcher
from fleet.manager import FleetManager
from planners.cbsh_planner import CBSHPlanner
from ml_predictor.predictor import EnergyTimePredictor
from environment import Environment, WeatherSystem
from utils.coordinate_manager import CoordinateManager
from server import update_simulation

@pytest.fixture
def test_dependencies():
    coord_manager = CoordinateManager()
    env = Environment(WeatherSystem(seed=123), coord_manager)
    predictor = EnergyTimePredictor()
    predictor.load_model()
    mock_planner = MagicMock(spec=CBSHPlanner)
    mock_planner.smoother = MagicMock()
    mock_planner.smoother.smooth_path.side_effect = lambda path, env: path
    mock_planner.env = env
    mock_lock = threading.Lock()
    fleet_manager = FleetManager(mock_planner, predictor, mock_lock)
    dispatcher = Dispatcher(fleet_manager)
    return {
        "state": get_initial_state(),
        "fleet_manager": fleet_manager,
        "dispatcher": dispatcher,
        "mock_planner": mock_planner,
        "planners": {"coord_manager": coord_manager, "env": env, "predictor": predictor}
    }

def test_full_mission_lifecycle(test_dependencies):
    state = test_dependencies['state']
    fm = test_dependencies['fleet_manager']
    dispatcher = test_dependencies['dispatcher']
    mock_planner = test_dependencies['mock_planner']
    planners = test_dependencies['planners']

    # Dispatch a single order from Hub A
    order = {'id': "Order1", 'pos': DESTINATIONS['NYU Campus'], 'payload_kg': 1.0, 'dest_name': 'NYU Campus'}
    hub_id = "HUB_A"
    status = dispatcher.dispatch_order(state, order, hub_id)
    assert status == "dispatched"
    assert not fm.planning_queue.empty()

    _, mission_obj = fm.planning_queue.queue[0]
    assigned_drone_id = mission_obj.drone_id
    
    # Configure mock planner for this specific mission
    mock_path = [mission_obj.start_pos] + [s['pos'] for s in mission_obj.stops] + mission_obj.destinations[-1:]
    mock_solution = {assigned_drone_id: [(p, i * 60) for i, p in enumerate(mock_path)]}
    mock_planner.plan_fleet.return_value = mock_solution
    
    # Plan and execute
    success, plan_results = fm.plan_pending_missions(state)
    assert success is True
    
    mission_id = plan_results['successful_mission_ids'][0]
    state['drones'][assigned_drone_id].update(plan_results['drone_updates'][assigned_drone_id])
    state['active_missions'][mission_id].update(plan_results['mission_updates'][mission_id])
    
    # Run sim to completion
    loop_count = 0
    while mission_id in state['active_missions'] and loop_count < 5000:
        update_simulation(state, planners)
        loop_count += 1
    
    assert loop_count < 5000, "Simulation timed out"
    assert state['drones'][assigned_drone_id]['status'] == 'RECHARGING'
    assert 'Order1' in state['completed_orders']