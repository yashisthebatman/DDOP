# FILE: tests/test_system_state.py
import pytest
import os
import numpy as np
import json
from tinydb import TinyDB, JSONStorage

# We need to import the module we are going to patch
import system_state
from system_state import NumpyJSONEncoder, DB_FILE, STATE_DOC_ID

@pytest.fixture
def mock_config(monkeypatch):
    """Mocks config constants used by the system_state module."""
    monkeypatch.setattr(system_state, 'HUBS', {"Test Hub": (1.0, 1.0, 1.0)})
    monkeypatch.setattr(system_state, 'DRONE_BATTERY_WH', 100.0)
    monkeypatch.setattr(system_state, 'DRONE_MAX_PAYLOAD_KG', 5.0)
    monkeypatch.setattr(system_state, 'MODEL_FILE_PATH', "test_model.joblib")

@pytest.fixture
def temp_db_file(tmp_path):
    """Creates a temporary database file for isolated test runs."""
    db_path = tmp_path / "test_state.json"
    # This is crucial: we modify the global variable in the module under test
    # so all its functions use our temporary file.
    original_db_file = system_state.DB_FILE
    system_state.DB_FILE = str(db_path)
    yield str(db_path)
    # Teardown: restore the original path after the test
    system_state.DB_FILE = original_db_file

def test_initial_state_creation(mock_config, temp_db_file):
    """Verify that load_state() creates a valid initial state file if one doesn't exist."""
    assert not os.path.exists(temp_db_file)
    state = system_state.load_state()

    assert os.path.exists(temp_db_file)
    assert 'drones' in state
    assert 'pending_orders' in state
    assert len(state['drones']) > 0
    # FIX: Assert that the initial state correctly has ZERO pending orders.
    assert len(state['pending_orders']) == 0
    assert state['drones']['Drone 1']['home_hub'] == "Test Hub"

def test_save_and_load_consistency(mock_config, temp_db_file):
    """Create a state, modify it, save, load, and assert the change is present."""
    state = system_state.load_state()
    state['drones']['Drone 1']['battery'] = 55.5
    state['log'].append("A new message")
    state['simulation_time'] = np.float64(123.45) # Use a numpy type

    system_state.save_state(state)
    new_state = system_state.load_state()

    assert new_state['drones']['Drone 1']['battery'] == 55.5
    assert new_state['log'][-1] == "A new message"
    # Verify that the numpy type was correctly serialized and deserialized
    assert isinstance(new_state['simulation_time'], float)
    assert new_state['simulation_time'] == pytest.approx(123.45)

def test_reset_state_file(mock_config, temp_db_file):
    """Test that reset clears the old file and creates a fresh initial state."""
    # Create and modify a state
    state = system_state.load_state()
    state['log'].append("This should be deleted.")
    system_state.save_state(state)

    # Reset the state
    reset_state = system_state.reset_state_file()

    # Assert the reset state is the initial one
    assert len(reset_state['log']) == 1
    assert reset_state['log'][0] == "System initialized. Add orders to begin."

    # Load again from file to be sure
    loaded_state = system_state.load_state()
    assert len(loaded_state['log']) == 1