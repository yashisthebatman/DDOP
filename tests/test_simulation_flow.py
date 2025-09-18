# FILE: tests/test_simulation_flow.py
import pytest
import time
import threading
import logging
from unittest.mock import MagicMock, patch

from config import HUBS, DESTINATIONS, DRONE_BATTERY_WH
from system_state import get_initial_state
from dispatch.dispatcher import Dispatcher, get_hub_by_id
from fleet.manager import FleetManager
from planners.cbsh_planner import CBSHPlanner
from ml_predictor.predictor import EnergyTimePredictor
from environment import Environment, WeatherSystem
from utils.coordinate_manager import CoordinateManager
from server import update_simulation
import simulation.event_injector as event_injector
from simulation.contingency_planner import _find_nearest_hub

@pytest.fixture
def test_dependencies():
    state = get_initial_state()
    coord_manager = CoordinateManager()
    env = Environment(WeatherSystem(seed=123), coord_manager)
    predictor = EnergyTimePredictor()
    predictor.load_model()
    mock_planner = MagicMock(spec=CBSHPlanner)
    mock_planner.smoother = MagicMock()
    mock_planner.smoother.smooth_path.side_effect = lambda path, env: path
    mock_planner.env = env

    # Mock planner needs to handle both normal and emergency missions
    def mock_plan_fleet(agents):
        solution = {}
        for agent in agents:
            mission_dict = next((m for m in state['active_missions'].values() if m['drone_id'] == agent.id), None)
            if not mission_dict: return None # Planner fails if no mission is found

            waypoints = [agent.start_pos]
            if mission_dict.get('stops'):
                waypoints.extend([stop['pos'] for stop in mission_dict['stops']])
            waypoints.append(agent.goal_pos)
            
            mock_path = [waypoints[0]]
            for p in waypoints[1:]:
                if p != mock_path[-1]: mock_path.append(p)

            solution[agent.id] = [(p, i * 60) for i, p in enumerate(mock_path)]
        return solution

    mock_planner.plan_fleet.side_effect = mock_plan_fleet
    
    with patch('simulation.contingency_planner.SingleAgentPlanner') as mock_sap:
        def mock_emergency_plan(start_pos, end_pos):
            logging.info(f"--- [MOCK] Mock emergency planner called for path from {start_pos} to {end_pos} ---")
            return ([start_pos, end_pos], "Success")
        mock_sap.return_value.find_strategic_path_rrt.side_effect = mock_emergency_plan
        
        mock_lock = threading.Lock()
        fleet_manager = FleetManager(mock_planner, predictor, mock_lock)
        dispatcher = Dispatcher(fleet_manager, predictor)

        yield {
            "state": state,
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

    order = {'id': "Order1", 'pos': DESTINATIONS['NYU Campus'], 'payload_kg': 1.0, 'dest_name': 'NYU Campus'}
    hub_id = "HUB_A"
    status = dispatcher.dispatch_order(state, order, hub_id)
    assert status == "dispatched"
    assert not fm.planning_queue.empty()

    _, mission_obj = fm.planning_queue.queue[0]
    assigned_drone_id = mission_obj.drone_id
    
    success, plan_results = fm.plan_pending_missions(state)
    assert success is True
    
    mission_id = plan_results['successful_mission_ids'][0]
    state['drones'][assigned_drone_id].update(plan_results['drone_updates'][assigned_drone_id])
    state['active_missions'][mission_id].update(plan_results['mission_updates'][mission_id])
    
    loop_count = 0
    while mission_id in state['active_missions'] and loop_count < 5000:
        update_simulation(state, planners)
        loop_count += 1
    
    assert loop_count < 5000, "Simulation timed out"
    assert state['drones'][assigned_drone_id]['status'] == 'RECHARGING'
    assert 'Order1' in state['completed_orders']

def test_failed_mission_lifecycle(test_dependencies):
    """
    Ensures that when a mission fails mid-flight due to a critical battery fault,
    the order is correctly requeued and the drone diverts to the NEAREST hub.
    """
    state = test_dependencies['state']
    fm = test_dependencies['fleet_manager']
    dispatcher = test_dependencies['dispatcher']
    planners = test_dependencies['planners']

    order = {'id': "OrderFail", 'pos': DESTINATIONS['StuyTown Apartments'], 'payload_kg': 1.0, 'dest_name': 'StuyTown Apartments'}
    hub_id = "HUB_A" # Start from a hub that is NOT the closest to the destination
    status = dispatcher.dispatch_order(state, order, hub_id)
    assert status == "dispatched"

    _, mission_obj = fm.planning_queue.queue[0]
    assigned_drone_id = mission_obj.drone_id
    
    success, plan_results = fm.plan_pending_missions(state)
    assert success is True
    mission_id = plan_results['successful_mission_ids'][0]
    state['drones'][assigned_drone_id].update(plan_results['drone_updates'][assigned_drone_id])
    state['active_missions'][mission_id].update(plan_results['mission_updates'][mission_id])

    # Let the simulation run until the drone is at the delivery location
    loop_count = 0
    while state['drones'][assigned_drone_id]['status'] != 'PERFORMING_DELIVERY' and loop_count < 1000:
        update_simulation(state, planners)
        loop_count += 1
    assert loop_count < 1000, "Drone never reached destination to start delivery"

    logging.info(f"--- [TEST] Injecting critical battery fault for {assigned_drone_id} at t={state['simulation_time']:.1f} ---")
    # Set battery to a value low enough to trigger emergency immediately, interrupting the delivery.
    state['drones'][assigned_drone_id]['battery'] = 2.0

    # The nearest hub at the DELIVERY LOCATION is Hub B, not the starting Hub A.
    nearest_hub_id_at_delivery = _find_nearest_hub(DESTINATIONS['StuyTown Apartments'], planners['coord_manager'])[0]
    assert nearest_hub_id_at_delivery == "HUB_B"
    
    # Run simulation to completion. The original mission ID will be deleted, so we loop until no active missions are left.
    loop_count = 0
    logging.info(f"--- [TEST] Resuming simulation to observe contingency handling ---")
    while state['active_missions'] and loop_count < 5000:
        update_simulation(state, planners)
        loop_count += 1
    logging.info(f"--- [TEST] Simulation completed at t={state['simulation_time']:.1f} ---")
        
    assert loop_count < 5000, "Simulation timed out"
    
    # 1. The delivery was interrupted, so the order should NOT be in completed_orders.
    assert "OrderFail" not in state['completed_orders']
    # 2. Because the mission failed before delivery completion, the order MUST be re-queued.
    assert "OrderFail" in state['pending_orders']
    
    final_drone_state = state['drones'][assigned_drone_id]
    assert final_drone_state['status'] == 'RECHARGING'
    
    # 3. The emergency was triggered at the delivery location, so the drone should have
    #    returned to the nearest hub, which is HUB_B.
    assert final_drone_state['home_hub'] == "HUB_B"
    
    assert len(state['completed_missions_log']) > 0
    failure_log = next((log for log in state['completed_missions_log'] if log['mission_id'] == mission_id), None)
    assert failure_log is not None, "Failure log for the original mission was not created."
    assert "Failed" in failure_log['outcome']