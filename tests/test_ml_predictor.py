# tests/test_ml_predictor.py

import pytest
import joblib
import numpy as np
from unittest.mock import MagicMock

from ml_predictor.predictor import EnergyTimePredictor

# ==============================================================================
# This new test file is crucial. It tests the logic of the EnergyTimePredictor
# class itself, including model loading, validation, and fallback mechanisms.
# These tests would have immediately caught the KeyError bug.
# ==============================================================================

# --- FIX: Create a simple, picklable class instead of using MagicMock for the model ---
class PicklableMockModel:
    def __init__(self, return_value):
        self._return_value = np.array([return_value])
    
    def predict(self, features):
        return self._return_value

@pytest.fixture
def mock_physics_predictor():
    """A mock of the physics predictor to ensure it's called on fallback."""
    predictor = MagicMock()
    # Define a specific return value to check against
    predictor.predict.return_value = (123.45, 67.89)
    return predictor

def create_mock_model_file(tmp_path, content):
    """Helper function to create a temporary model file for testing."""
    model_dir = tmp_path / "ml_predictor"
    model_dir.mkdir()
    model_path = model_dir / "test_model.joblib"
    joblib.dump(content, model_path)
    return str(model_path)

def test_predictor_loads_valid_model_successfully(tmp_path):
    """
    Tests that the predictor correctly loads a model file with the expected
    dictionary structure.
    """
    # Use our new picklable mock class
    mock_time_model = PicklableMockModel(return_value=10.0)
    mock_energy_model = PicklableMockModel(return_value=20.0)

    valid_model_dict = {
        'time_model': mock_time_model,
        'energy_model': mock_energy_model
    }
    
    model_path = create_mock_model_file(tmp_path, valid_model_dict)
    
    # Initialize the predictor with the path to our valid mock model
    predictor = EnergyTimePredictor(model_path=model_path)
    
    # Assert that the models were loaded and not the fallback
    assert predictor.models is not None
    
    # Make a prediction and verify it uses the mock ML models
    time, energy = predictor.predict((0,0,0), (1,1,1), 1, np.zeros(3))
    assert time == 10.0
    assert energy == 20.0

def test_predictor_falls_back_if_model_file_is_not_dict(tmp_path, mock_physics_predictor):
    """
    Tests the exact failure case you experienced: the file contains a raw
    object, not a dictionary.
    """
    # Create a file containing just a raw object (simulating the bug)
    invalid_model_content = PicklableMockModel(return_value=0)
    model_path = create_mock_model_file(tmp_path, invalid_model_content)
    
    # Inject our mock physics predictor to see if it gets used
    predictor = EnergyTimePredictor(model_path=model_path)
    predictor.fallback_predictor = mock_physics_predictor

    # Assert that the ML models were NOT loaded
    assert predictor.models is None
    
    # Make a prediction and verify it uses the fallback
    time, energy = predictor.predict((0,0,0), (1,1,1), 1, np.zeros(3))
    assert time == 123.45  # The specific value from our mock fallback
    assert energy == 67.89
    mock_physics_predictor.predict.assert_called_once()
    
def test_predictor_falls_back_if_model_file_missing_keys(tmp_path, mock_physics_predictor):
    """
    Tests that the predictor falls back if the dictionary is missing
    the required 'time_model' or 'energy_model' keys.
    """
    invalid_dict = {'some_other_key': PicklableMockModel(return_value=0)}
    model_path = create_mock_model_file(tmp_path, invalid_dict)

    predictor = EnergyTimePredictor(model_path=model_path)
    predictor.fallback_predictor = mock_physics_predictor

    assert predictor.models is None
    
    time, energy = predictor.predict((0,0,0), (1,1,1), 1, np.zeros(3))
    assert time == 123.45
    assert energy == 67.89
    mock_physics_predictor.predict.assert_called_once()

def test_predictor_falls_back_if_model_file_not_found(mock_physics_predictor):
    """
    Tests that the predictor handles a non-existent model file gracefully.
    """
    non_existent_path = "path/that/does/not/exist.joblib"
    
    predictor = EnergyTimePredictor(model_path=non_existent_path)
    predictor.fallback_predictor = mock_physics_predictor

    assert predictor.models is None
    
    time, energy = predictor.predict((0,0,0), (1,1,1), 1, np.zeros(3))
    assert time == 123.45
    assert energy == 67.89
    mock_physics_predictor.predict.assert_called_once()