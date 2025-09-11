import pytest
from unittest.mock import MagicMock
from fleet.manager import FleetManager, Mission
from fleet.cbs_components import Agent

@pytest.fixture
def mock_cbs_planner():
    planner = MagicMock()
    planner.coord_manager = MagicMock()
    planner.coord_manager.world_to_local_grid.side_effect = lambda p: (int(p[0]), int(p[1]), int(p[2]))
    return planner

@pytest.fixture
def fleet_manager(mock_cbs_planner):
    # FIX: The FleetManager constructor requires a 'predictor' argument.
    # A mock predictor is now provided to prevent the TypeError during setup.
    predictor_mock = MagicMock()
    return FleetManager(mock_cbs_planner, predictor_mock)

def test_add_mission(fleet_manager):
    mission = Mission("drone1", (0,0,10), [(10,10,10)], 1.0, {})
    fleet_manager.add_mission(mission)
    assert "drone1" in fleet_manager.missions
    assert fleet_manager.missions["drone1"].state == "PENDING"

def test_planning_cycle_converts_missions_to_agents(fleet_manager):
    """
    Tests that the manager correctly creates Agent objects for the planner
    based on the current mission legs.
    """
    m1 = Mission("d1", (0,0,10), [(10,10,10)], 1.0, {'w_time': 1})
    m2 = Mission("d2", (20,0,10), [(20,20,10), (30,30,10)], 2.0, {'w_time': 0})
    fleet_manager.add_mission(m1)
    fleet_manager.add_mission(m2)

    fleet_manager.cbs_planner.plan_fleet.return_value = {} # Mock successful plan
    
    fleet_manager.execute_planning_cycle()

    # Assert that plan_fleet was called with the correct Agent objects
    call_args = fleet_manager.cbs_planner.plan_fleet.call_args[0][0]
    assert len(call_args) == 2
    
    agent1 = next(a for a in call_args if a.id == "d1")
    agent2 = next(a for a in call_args if a.id == "d2")

    assert agent1.start_pos == (0,0,10)
    assert agent1.goal_pos == (10,10,10)
    assert agent1.config['payload_kg'] == 1.0
    assert agent1.config['w_time'] == 1

    assert agent2.start_pos == (20,0,10)
    assert agent2.goal_pos == (20,20,10)
    assert agent2.config['payload_kg'] == 2.0
    assert agent2.config['w_time'] == 0

def test_multi_stop_advances_to_next_leg(fleet_manager):
    """
    Tests that after completing a leg, the manager correctly sets up
    the next leg for the subsequent planning cycle.
    """
    mission = Mission("drone1", (0,0,10), [(10,10,10), (20,20,10)], 1.0, {})
    fleet_manager.add_mission(mission)
    
    # Simulate first leg completion
    mission.advance_leg()
    
    assert mission.current_leg_index == 1
    start, end = mission.get_current_leg()
    assert start == (10,10,10) # Start of leg 2 is end of leg 1
    assert end == (20,20,10)
    assert mission.state == "PENDING" # Ready for next plan

    # Simulate second leg completion
    mission.advance_leg()
    assert mission.is_complete()
    assert mission.state == "COMPLETED"