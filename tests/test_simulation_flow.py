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
        mock_sap.return_value.find_strategic_path_rrt.return_value = ([(-74.0, 40.72, 50), HUBS[1]['location']], "Success")
        
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

# NEW: Test for the complete failure and recovery cycle
def test_failed_mission_lifecycle(test_dependencies):
    """
    Ensures that when a mission fails mid-flight, the order is requeued,
    the failure is logged, and the drone correctly returns to the NEAREST hub.
    """
    state = test_dependencies['state']
    fm = test_dependencies['fleet_manager']
    dispatcher = test_dependencies['dispatcher']
    planners = test_dependencies['planners']

    # Dispatch an order from Hub A to a far destination
    order = {'id': "OrderFail", 'pos': DESTINATIONS['StuyTown Apartments'], 'payload_kg': 1.0, 'dest_name': 'StuyTown Apartments'}
    hub_id = "HUB_A"
    status = dispatcher.dispatch_order(state, order, hub_id)
    assert status == "dispatched"

    _, mission_obj = fm.planning_queue.queue[0]
    assigned_drone_id = mission_obj.drone_id
    
    success, plan_results = fm.plan_pending_missions(state)
    assert success is True
    mission_id = plan_results['successful_mission_ids'][0]
    state['drones'][assigned_drone_id].update(plan_results['drone_updates'][assigned_drone_id])
    state['active_missions'][mission_id].update(plan_results['mission_updates'][mission_id])

    # Let the simulation run for a bit to get the drone mid-air
    for _ in range(30):
        update_simulation(state, planners)
    
    # Manually inject a battery fault to trigger the contingency
    logging.info(f"\n\n[TEST_LOG] >>> INJECTING FAULT at t={state['simulation_time']:.1f}. Setting battery for {assigned_drone_id} to 10.0Wh. Position: {state['drones'][assigned_drone_id]['pos']}\n\n")
    state['drones'][assigned_drone_id]['battery'] = 10.0

    # The nearest hub to StuyTown is Hub B, not the original Hub A
    nearest_hub_id, _ = _find_nearest_hub(DESTINATIONS['StuyTown Apartments'], planners['coord_manager'])
    assert nearest_hub_id == "HUB_B"
    
    # Run simulation to completion
    loop_count = 0
    logging.info(f"\n[TEST_LOG] >>> STARTING FINAL SIMULATION LOOP <<<\n")
    while state['active_missions'] and loop_count < 5000:
        # IMPLANTED LOGGING
        drone_state = state['drones'][assigned_drone_id]
        logging.info(
            f"[TEST_TICK] t={state['simulation_time']:.1f}, "
            f"Drone: {drone_state['id']}, "
            f"Status: {drone_state['status']}, "
            f"Battery: {drone_state['battery']:.2f}Wh, "
            f"Pos: ({drone_state['pos'][0]:.4f}, {drone_state['pos'][1]:.4f}, {drone_state['pos'][2]:.1f})"
        )
        update_simulation(state, planners)
        loop_count += 1
    logging.info(f"\n[TEST_LOG] >>> FINAL SIMULATION LOOP ENDED after {loop_count} ticks. Final sim time: {state['simulation_time']:.1f} <<<\n")
        
    assert loop_count < 5000, "Simulation timed out"
    
    # Verify the entire recovery process
    assert "OrderFail" not in state['completed_orders']
    assert "OrderFail" in state['pending_orders']
    
    final_drone_state = state['drones'][assigned_drone_id]
    assert final_drone_state['status'] == 'RECHARGING'
    assert final_drone_state['home_hub'] == nearest_hub_id # Critical check: went to nearest hub
    
    assert len(state['completed_missions_log']) > 0
    failure_log = next(log for log in state['completed_missions_log'] if log['mission_id'] == mission_id)
    assert "Failed" in failure_log['outcome']