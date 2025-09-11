# FILE: tests/test_ml_predictor.py
import pytest
import joblib
import numpy as np
import os
from unittest.mock import MagicMock, patch

from ml_predictor.predictor import EnergyTimePredictor
from config import MODEL_FILE_PATH

# A simple mock model class that can be pickled by joblib
class MockModel:
    def __init__(self, return_value):
        self._return_value = np.array([return_value])
    def predict(self, features):
        return self._return_value

@pytest.fixture
def mock_physics_predictor():
    predictor = MagicMock()
    predictor.predict.return_value = (123.45, 67.89)
    return predictor

def test_predictor_loads_from_local_file(tmp_path, mock_physics_predictor):
    """Tests that the predictor correctly loads a model from a local joblib file."""
    # Create a dummy model file in the temporary test directory
    model_dir = tmp_path / "ml_predictor"
    model_dir.mkdir()
    model_path = model_dir / os.path.basename(MODEL_FILE_PATH)

    mock_time_model = MockModel(10.0)
    mock_energy_model = MockModel(20.0)
    valid_model_dict = {'time_model': mock_time_model, 'energy_model': mock_energy_model}
    joblib.dump(valid_model_dict, model_path)
    
    # Use patch to point the config constant to our temporary file
    with patch('ml_predictor.predictor.MODEL_FILE_PATH', str(model_path)):
        predictor = EnergyTimePredictor()
        predictor.fallback_predictor = mock_physics_predictor
        
        assert predictor.models is not None
        
        time, energy = predictor.predict((0,0,0), (1,1,1), 1, np.zeros(3))
        assert time == 10.0
        assert energy == 20.0
        mock_physics_predictor.predict.assert_not_called()

def test_predictor_falls_back_if_file_not_found(tmp_path, mock_physics_predictor):
    """Tests that the predictor falls back gracefully if the model file does not exist."""
    # Point the config to a file that doesn't exist in the temp directory
    non_existent_path = tmp_path / "non_existent_model.joblib"
    
    with patch('ml_predictor.predictor.MODEL_FILE_PATH', str(non_existent_path)):
        predictor = EnergyTimePredictor()
        predictor.fallback_predictor = mock_physics_predictor

        assert predictor.models is None
        
        time, energy = predictor.predict((0,0,0), (1,1,1), 1, np.zeros(3))
        assert time == 123.45
        assert energy == 67.89
        mock_physics_predictor.predict.assert_called_once()