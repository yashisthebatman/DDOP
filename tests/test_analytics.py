# FILE: tests/test_analytics.py

import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import MagicMock # <--- FIX: Added the missing import

# This test file now imports the REAL update_simulation to avoid code duplication
# and ensure it's testing the actual application logic.
from server import update_simulation 
from system_state import get_initial_state
from config import DRONE_BATTERY_WH, DRONE_RECHARGE_TIME_S

# A helper function to represent the KPI calculation logic
def calculate_kpis(log_df):
    if log_df.empty:
        return 0, 0, 0, 0
    
    # On-Time Delivery Rate
    completed_df = log_df[log_df['outcome'] == 'Completed']
    on_time = (completed_df['actual_duration_sec'] <= completed_df['planned_duration_sec'] * 1.05).sum() # 5% buffer
    on_time_rate = (on_time / len(completed_df)) * 100 if len(completed_df) > 0 else 0
    
    # Energy Prediction Accuracy (Average % Error)
    valid_energy_df = completed_df[completed_df['planned_energy_wh'] > 0]
    if not valid_energy_df.empty:
        energy_error = (abs(valid_energy_df['actual_energy_wh'] - valid_energy_df['planned_energy_wh']) / valid_energy_df['planned_energy_wh']).mean() * 100
    else:
        energy_error = 0.0
    if pd.isna(energy_error): energy_error = 0.0

    total_missions = len(log_df)
    failure_rate = (log_df['outcome'] != 'Completed').sum() / total_missions * 100 if total_missions > 0 else 0
    
    return on_time_rate, energy_error, total_missions, failure_rate

def test_mission_log_creation_on_completion():
    """Simulate a mission completion and assert the log entry is correct."""
    state = get_initial_state()
    state['simulation_time'] = 50.0
    
    mission_id = "M-123"
    drone_id = "Drone 1"
    
    state['drones'][drone_id]['status'] = 'EN ROUTE'
    state['drones'][drone_id]['mission_id'] = mission_id
    state['drones'][drone_id]['battery'] = 155.0 # Start battery 200, planned energy 45. Should be 155 at end.

    state['active_missions'][mission_id] = {
        'mission_id': mission_id,
        'drone_id': drone_id,
        'order_ids': ['Order1'],
        'start_time': 10.0,
        'total_planned_time': 40.5, # This will make it complete on the next tick
        'total_planned_energy': 45.0,
        'path_world_coords': [(-74.0, 40.7, 50), (-74.01, 40.71, 60)],
        'destinations': [(-74.01, 40.71, 60)],
        'start_battery': DRONE_BATTERY_WH,
        'mission_time_elapsed': 40.0, # Almost done
        'flight_time_elapsed': 40.0,
        'total_maneuver_time': 0,
        'stops': []
    }
    
    # Mock planners dict as it's needed by update_simulation
    mock_planners = {"coord_manager": MagicMock()}
    
    update_simulation(state, mock_planners) # This tick should complete the mission
    
    assert mission_id not in state['active_missions']
    assert len(state['completed_missions_log']) == 1
    
    log = state['completed_missions_log'][0]
    assert log['mission_id'] == mission_id
    assert log['drone_id'] == drone_id
    assert log['completion_timestamp'] == 50.5 # 50.0 + 0.5 step
    assert log['planned_duration_sec'] == 40.5
    assert log['actual_duration_sec'] == 40.5 # 50.5 - 10.0
    assert log['planned_energy_wh'] == 45.0
    assert log['actual_energy_wh'] == pytest.approx(45.0)
    assert log['outcome'] == 'Completed'

def test_kpi_calculations():
    """Create a mock log and test the KPI calculation logic."""
    mock_log_data = [
        {'outcome': 'Completed', 'planned_duration_sec': 100, 'actual_duration_sec': 90, 'planned_energy_wh': 50, 'actual_energy_wh': 52},
        {'outcome': 'Completed', 'planned_duration_sec': 100, 'actual_duration_sec': 110, 'planned_energy_wh': 50, 'actual_energy_wh': 60},
        {'outcome': 'Completed', 'planned_duration_sec': 100, 'actual_duration_sec': 100, 'planned_energy_wh': 50, 'actual_energy_wh': 48},
        {'outcome': 'Failed: Low Battery', 'planned_duration_sec': 120, 'actual_duration_sec': 80, 'planned_energy_wh': 60, 'actual_energy_wh': 40},
    ]
    df = pd.DataFrame(mock_log_data)
    
    on_time_rate, energy_error, total_missions, failure_rate = calculate_kpis(df)
    
    # On-time allows for a 5% buffer. 90 and 100 are on time. 110 is not.
    assert on_time_rate == pytest.approx((2/3) * 100) 
    # Energy error is only calculated on completed missions
    assert energy_error == pytest.approx(((abs(52-50)/50 + abs(60-50)/50 + abs(48-50)/50) / 3) * 100)
    assert total_missions == 4
    assert failure_rate == pytest.approx(25.0)