# FILE: tests/test_analytics.py

import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import MagicMock 

from server import update_simulation 
from system_state import get_initial_state
from config import DRONE_BATTERY_WH, DRONE_RECHARGE_TIME_S, HUBS
from utils.coordinate_manager import CoordinateManager

# Helper to get hub by ID
def get_hub_by_id(hub_id):
    return next((h for h in HUBS if h['id'] == hub_id), None)

def calculate_kpis(log_df):
    if log_df.empty: return 0, 0, 0, 0
    completed_df = log_df[log_df['outcome'] == 'Completed']
    if completed_df.empty: return 0, 0, len(log_df), 100.0
    on_time = (completed_df['actual_duration_sec'] <= completed_df['planned_duration_sec'] * 1.05).sum()
    on_time_rate = (on_time / len(completed_df)) * 100
    valid_energy_df = completed_df[completed_df['planned_energy_wh'] > 0]
    energy_error = (abs(valid_energy_df['actual_energy_wh'] - valid_energy_df['planned_energy_wh']) / valid_energy_df['planned_energy_wh']).mean() * 100 if not valid_energy_df.empty else 0.0
    if pd.isna(energy_error): energy_error = 0.0
    total_missions = len(log_df)
    failure_rate = (log_df['outcome'] != 'Completed').sum() / total_missions * 100
    return on_time_rate, energy_error, total_missions, failure_rate

def test_mission_log_creation_on_completion():
    """Simulate a mission completion and assert the log entry is correct."""
    state = get_initial_state()
    state['simulation_time'] = 0.0
    
    mission_id, drone_id = "M-123", "Drone 1"
    
    state['drones'][drone_id].update({'status': 'EN ROUTE', 'mission_id': mission_id, 'pos': (-74.0, 40.7, 50)})
    
    # FIX: Get hub location from new list format
    end_hub_obj = get_hub_by_id("HUB_A")
    end_hub_pos = end_hub_obj['location']
    
    state['active_missions'][mission_id] = {
        'mission_id': mission_id, 'drone_id': drone_id, 'order_ids': ['Order1'], 'start_time': 0.0,
        'total_planned_time': 200.0, 'total_planned_energy': 45.0, 'path_world_coords': [(-74.0, 40.7, 50), end_hub_pos],
        'destinations': [end_hub_pos], 'start_battery': DRONE_BATTERY_WH, 'mission_time_elapsed': 0.0,
        'flight_time_elapsed': 0.0, 'total_maneuver_time': 0, 'stops': [], 'current_stop_index': 0,
        'end_hub': end_hub_obj['id']
    }
    
    # MODIFIED: Create a more complete mock planners dict to satisfy the contingency checker.
    env_mock = MagicMock()
    env_mock.was_nfz_just_added = False
    predictor_mock = MagicMock()
    predictor_mock.predict.return_value = (10.0, 5.0) # a default safe value
    mock_planners = {
        "coord_manager": CoordinateManager(),
        "env": env_mock,
        "predictor": predictor_mock
    }
    
    loop_count = 0
    while mission_id in state['active_missions'] and loop_count < 1000:
        update_simulation(state, mock_planners)
        loop_count += 1
    
    assert loop_count < 1000
    assert mission_id not in state['active_missions']
    assert len(state['completed_missions_log']) == 1
    assert state['completed_missions_log'][0]['outcome'] == 'Completed'

def test_kpi_calculations():
    mock_log_data = [
        {'outcome': 'Completed', 'planned_duration_sec': 100, 'actual_duration_sec': 90, 'planned_energy_wh': 50, 'actual_energy_wh': 52},
        {'outcome': 'Completed', 'planned_duration_sec': 100, 'actual_duration_sec': 110, 'planned_energy_wh': 50, 'actual_energy_wh': 60},
        {'outcome': 'Completed', 'planned_duration_sec': 100, 'actual_duration_sec': 100, 'planned_energy_wh': 50, 'actual_energy_wh': 48},
        {'outcome': 'Failed: Low Battery', 'planned_duration_sec': 120, 'actual_duration_sec': 80, 'planned_energy_wh': 60, 'actual_energy_wh': 40},
    ]
    df = pd.DataFrame(mock_log_data)
    on_time_rate, energy_error, total_missions, failure_rate = calculate_kpis(df)
    assert on_time_rate == pytest.approx((2/3) * 100) 
    assert energy_error == pytest.approx(((abs(52-50)/50 + abs(60-50)/50 + abs(48-50)/50) / 3) * 100)
    assert total_missions == 4
    assert failure_rate == pytest.approx(25.0)