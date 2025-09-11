import pytest
import joblib
import numpy as np
from unittest.mock import MagicMock, patch

from ml_predictor.predictor import EnergyTimePredictor

class PicklableMockModel:
    def __init__(self, return_value):
        self._return_value = np.array([return_value])
    def predict(self, features):
        return self._return_value

@pytest.fixture
def mock_physics_predictor():
    predictor = MagicMock()
    predictor.predict.return_value = (123.45, 67.89)
    return predictor

def create_mock_model_file(tmp_path, content):
    model_path = tmp_path / "test_model.joblib"
    joblib.dump(content, model_path)
    return str(model_path)

@patch('ml_predictor.predictor.mlflow')
def test_predictor_loads_from_mock_mlflow(mock_mlflow, mock_physics_predictor):
    """Tests that the predictor correctly loads a model from a mocked MLflow registry."""
    mock_time_model = PicklableMockModel(return_value=10.0)
    mock_energy_model = PicklableMockModel(return_value=20.0)
    valid_model_dict = {'time_model': mock_time_model, 'energy_model': mock_energy_model}
    
    mock_mlflow.sklearn.load_model.return_value = valid_model_dict
    
    predictor = EnergyTimePredictor()
    predictor.fallback_predictor = mock_physics_predictor
    
    assert predictor.models is not None
    mock_mlflow.sklearn.load_model.assert_called_once_with("models:/drone-energy-time-predictor/Production")
    
    time, energy = predictor.predict((0,0,0), (1,1,1), 1, np.zeros(3))
    assert time == 10.0
    assert energy == 20.0
    mock_physics_predictor.predict.assert_not_called()

@patch('ml_predictor.predictor.mlflow')
def test_predictor_falls_back_if_mlflow_fails(mock_mlflow, mock_physics_predictor):
    """Tests that the predictor falls back gracefully if MLflow throws an error."""
    mock_mlflow.sklearn.load_model.side_effect = Exception("Registry not available")
    
    predictor = EnergyTimePredictor()
    predictor.fallback_predictor = mock_physics_predictor

    assert predictor.models is None
    
    time, energy = predictor.predict((0,0,0), (1,1,1), 1, np.zeros(3))
    assert time == 123.45
    assert energy == 67.89
    mock_physics_predictor.predict.assert_called_once()