# FILE: tests/test_retrainer.py

import pytest
import pandas as pd
import os
from unittest.mock import patch, MagicMock

from training.retrainer import retrain_model, get_next_model_version_path

@pytest.fixture
def setup_test_data_files(tmp_path):
    """Creates dummy data files for testing the retrainer."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    
    initial_data_path = training_dir / "training_data.csv"
    real_world_data_path = data_dir / "real_world_flight_segments.csv"
    
    # Create some dummy data that matches expected columns
    columns = ['distance_3d', 'payload_kg', 'actual_time', 'actual_energy', 'altitude_change', 'horizontal_distance', 'wind_speed', 'wind_alignment', 'turning_angle', 'p1_alt', 'p2_alt', 'abs_alt_change']
    pd.DataFrame([[100, 1, 10, 20, 10, 100, 5, 0.5, 0, 50, 60, 10]], columns=columns).to_csv(initial_data_path, index=False)
    pd.DataFrame([[200, 2, 20, 40, 20, 200, 2, -0.2, 10, 60, 80, 20]], columns=columns).to_csv(real_world_data_path, index=False)
    
    return initial_data_path, real_world_data_path

def test_data_is_appended(setup_test_data_files, monkeypatch):
    """Check that the retrainer script correctly combines old and new data."""
    initial_path, real_world_path = setup_test_data_files
    
    # Mock the training function itself so we only test the data loading part
    mock_run_training = MagicMock(return_value=(True, {}))
    monkeypatch.setattr('training.retrainer.run_training_on_dataframe', mock_run_training)
    
    # Patch config paths to point to our temp files
    monkeypatch.setattr('training.retrainer.TRAINING_DATA_PATH', str(initial_path))
    monkeypatch.setattr('training.retrainer.REAL_WORLD_DATA_PATH', str(real_world_path))
    
    # Mock system_state to avoid file IO
    # FIX: Add the 'log' key to the mock state
    mock_state = {'active_model_path': 'model.joblib', 'log': []}
    monkeypatch.setattr('training.retrainer.system_state.load_state', lambda: mock_state)
    monkeypatch.setattr('training.retrainer.system_state.save_state', lambda state: None)

    retrain_model()

    # Assert that the training function was called with the combined data
    mock_run_training.assert_called_once()
    call_args, _ = mock_run_training.call_args
    combined_df = call_args[0]
    
    assert len(combined_df) == 2 # 1 row from initial, 1 from new
    assert combined_df.iloc[0]['distance_3d'] == 100
    assert combined_df.iloc[1]['distance_3d'] == 200

def test_state_is_updated_after_retraining(monkeypatch):
    """Assert that the model path in the state file is correctly updated."""
    # Mock data loading
    monkeypatch.setattr('training.retrainer.pd.read_csv', lambda path: pd.DataFrame([['dummy']]*12, index=['distance_3d', 'payload_kg', 'actual_time', 'actual_energy', 'altitude_change', 'horizontal_distance', 'wind_speed', 'wind_alignment', 'turning_angle', 'p1_alt', 'p2_alt', 'abs_alt_change']).T)
    # Mock training process
    monkeypatch.setattr('training.retrainer.run_training_on_dataframe', MagicMock(return_value=(True, {})))
    
    # Mock system_state and capture the saved state
    saved_state = {}
    def mock_save(state):
        nonlocal saved_state
        saved_state = state.copy()
        
    initial_state = {'active_model_path': 'ml_predictor/drone_predictor_model_v2.joblib', 'log': []}
    monkeypatch.setattr('training.retrainer.system_state.load_state', lambda: initial_state)
    monkeypatch.setattr('training.retrainer.system_state.save_state', mock_save)

    retrain_model()
    
    # Check that the saved state has the incremented model path
    assert 'active_model_path' in saved_state
    assert saved_state['active_model_path'] == 'ml_predictor/drone_predictor_model_v3.joblib'
    assert "Model retrained" in saved_state['log'][0]

def test_get_next_model_version_path():
    assert get_next_model_version_path("model.joblib") == "model_v2.joblib"
    assert get_next_model_version_path("model_v1.joblib") == "model_v2.joblib"
    assert get_next_model_version_path("model_v10.joblib") == "model_v11.joblib"
    assert get_next_model_version_path("path/to/model_v3.pkl") == "path/to/model_v4.pkl"