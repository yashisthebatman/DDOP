# FILE: tests/test_e2e.py
import pytest
import time
import threading
from unittest.mock import MagicMock, patch

from system_state import get_initial_state
from server import update_simulation
from config import HUBS
from dispatch.dispatcher import Dispatcher, get_hub_by_id
from fleet.manager import FleetManager, Mission
from planners.cbsh_planner import CBSHPlanner
from ml_predictor.predictor import EnergyTimePredictor
from environment import Environment, WeatherSystem
from utils.coordinate_manager import CoordinateManager
from demand_analyzer import DemandAnalyzer

@pytest.fixture
def full_system_components():
    """Provides a complete set of real, integrated components for E2E testing."""
    state = get_initial_state()
    coord_manager = CoordinateManager()
    env = Environment(WeatherSystem(seed=42), coord_manager)
    predictor = EnergyTimePredictor()
    predictor.load_model()
    
    mock_lock = threading.Lock()
    mock_planner = MagicMock(spec=CBSHPlanner)
    
    def mock_plan_fleet(agents):
        solution = {}
        for agent in agents:
            mission_dict = next(m for m in state['active_missions'].values() if m['drone_id'] == agent.id)
            waypoints = [agent.start_pos]
            if mission_dict.get('stops'):
                waypoints.extend([stop['pos'] for stop in mission_dict['stops']])
            waypoints.append(agent.goal_pos)
            
            mock_path = [waypoints[0]]
            for point in waypoints[1:]:
                if point != mock_path[-1]:
                    mock_path.append(point)
            solution[agent.id] = [(p, i * 60) for i, p in enumerate(mock_path)]
        return solution
    mock_planner.plan_fleet.side_effect = mock_plan_fleet
    
    mock_planner.smoother = MagicMock()
    mock_planner.smoother.smooth_path.side_effect = lambda path, env: path
    mock_planner.env = env
    
    fleet_manager = FleetManager(mock_planner, predictor, mock_lock)
    # FIX: Pass the predictor to the Dispatcher constructor
    dispatcher = Dispatcher(fleet_manager, predictor)
    
    return {
        "state": state,
        "dispatcher": dispatcher,
        "fleet_manager": fleet_manager,
        "planners": {"coord_manager": coord_manager, "env": env, "predictor": predictor}
    }

@pytest.mark.slow
def test_e2e_full_workflow(full_system_components):
    """
    A thorough E2E test covering:
    1. Normal dispatch with smart selection.
    2. Out-of-range order rejection.
    3. Demand-based fleet rebalancing.
    """
    state = full_system_components['state']
    dispatcher = full_system_components['dispatcher']
    fm = full_system_components['fleet_manager']
    planners = full_system_components['planners']
    
    hub_a_id = "HUB_A"
    hub_b_id = "HUB_B"
    hub_a_loc = get_hub_by_id(hub_a_id)['location']

    # --- 1. Test Smart Dispatch ---
    best_drone_id = "Drone 1"
    state['drones'][best_drone_id]['battery'] = 200.0
    state['drones'][best_drone_id]['battery_health'] = 99.9
    for i in range(2,6):
        state['drones'][f'Drone {i}']['battery_health'] = 90.0
    state['drones']["Drone 2"]['battery'] = 150.0 
    
    order1 = {'id': 'Order-E2E-1', 'pos': (hub_a_loc[0] + 0.01, hub_a_loc[1], 50), 'payload_kg': 1.0}
    status = dispatcher.dispatch_order(state, order1, hub_a_id)
    assert status == "dispatched"
    
    _, mission_obj = fm.planning_queue.get()
    assert mission_obj.drone_id == best_drone_id
    
    fm.planning_queue.put((1, mission_obj)) 
    success, results = fm.plan_pending_missions(state)
    assert success is True
    state['drones'][best_drone_id].update(results['drone_updates'][best_drone_id])
    state['active_missions'][mission_obj.mission_id].update(results['mission_updates'][mission_obj.mission_id])

    # --- 2. Test Out of Range ---
    for i in range(6, 11):
        state['drones'][f'Drone {i}']['battery'] = 1.0
    
    order2 = {'id': 'Order-E2E-2', 'pos': (hub_a_loc[0] + 0.05, hub_a_loc[1], 50), 'payload_kg': 1.0}
    status = dispatcher.dispatch_order(state, order2, hub_b_id)
    assert status == "out_of_range"

    # --- 3. Test Fleet Rebalancing ---
    shutdown_event = threading.Event()
    analyzer = DemandAnalyzer(state, dispatcher, threading.Lock(), shutdown_event)
    for i in range(1, 6):
        if state['drones'][f'Drone {i}']['status'] != 'EN ROUTE':
            state['drones'][f'Drone {i}']['status'] = 'IDLE'
    for i in range(6, 11):
        state['drones'][f'Drone {i}']['status'] = 'EN ROUTE'
    analyzer.check_and_rebalance_fleet() 
    
    assert not fm.planning_queue.empty()
    _, rebalance_mission = fm.planning_queue.get()
    assert rebalance_mission.mission_id.startswith("REBALANCE")
    assert rebalance_mission.start_hub == hub_a_id
    assert rebalance_mission.end_hub == hub_b_id
    
    fm.planning_queue.put((1, rebalance_mission))
    success, rebalance_results = fm.plan_pending_missions(state)
    assert success is True
    rebalance_drone_id = rebalance_mission.drone_id
    state['drones'][rebalance_drone_id].update(rebalance_results['drone_updates'][rebalance_drone_id])
    state['active_missions'][rebalance_mission.mission_id].update(rebalance_results['mission_updates'][rebalance_mission.mission_id])
    
    # --- 4. Run Simulation to Completion ---
    active_mission_ids = {mission_obj.mission_id, rebalance_mission.mission_id}
    loop_count = 0
    while any(mid in state['active_missions'] for mid in active_mission_ids) and loop_count < 5000:
        update_simulation(state, planners)
        loop_count += 1
    assert loop_count < 5000

    assert state['drones'][best_drone_id]['home_hub'] == hub_a_id
    assert 'Order-E2E-1' in state['completed_orders']
    
    assert state['drones'][rebalance_drone_id]['home_hub'] == hub_b_id