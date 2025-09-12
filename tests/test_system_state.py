# FILE: tests/test_system_state.py
# --- PLEASE REPLACE THE ENTIRE FILE WITH THIS CODE ---

import pytest
import os
# We need to import the module we are going to patch
import system_state

@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    """
    Mocks dependencies to ensure tests are isolated and predictable.
    This uses a more direct way of patching the function.
    """
    # Patch the constants used by the system_state module
    monkeypatch.setattr(system_state, 'HUBS', {"Test Hub": (1, 1, 1)})
    monkeypatch.setattr(system_state, 'DESTINATIONS', {"Test Dest": (2, 2, 2)})
    monkeypatch.setattr(system_state, 'DRONE_BATTERY_WH', 100.0)
    monkeypatch.setattr(system_state, 'DRONE_MAX_PAYLOAD_KG', 5.0)

    # Define a simple function that has the correct signature (accepts low and high)
    def mock_uniform_function(low, high):
        return 2.5

    # Directly replace the 'uniform' function on numpy's 'random' object
    # This is a more robust way to mock this specific function call.
    monkeypatch.setattr(system_state.np.random, 'uniform', mock_uniform_function)


def test_initial_state_creation(tmp_path):
    """Verify that load_state() creates a valid initial state file if one doesn't exist."""
    db_file = tmp_path / "test_db.json"
    # Point the module's global DB_FILE variable to our temporary test file
    system_state.DB_FILE = str(db_file)

    assert not os.path.exists(db_file)
    state = system_state.load_state()

    assert os.path.exists(db_file)
    assert 'drones' in state
    assert 'pending_orders' in state
    assert state['pending_orders']['Test Dest']['payload_kg'] == 2.5 # Check mocked numpy


def test_save_and_load_consistency(tmp_path):
    """Create a state, modify it, save, load, and assert the change is present."""
    db_file = tmp_path / "test_db.json"
    system_state.DB_FILE = str(db_file)

    state = system_state.load_state()
    state['drones']['Drone 1']['battery'] = 55.5
    state['log'].append("A new message")

    system_state.save_state(state)
    new_state = system_state.load_state()

    assert new_state['drones']['Drone 1']['battery'] == 55.5
    assert new_state['log'][-1] == "A new message"


def test_add_order_persistence(tmp_path):
    """Add an order, save, load, and verify the order exists."""
    db_file = tmp_path / "test_db.json"
    system_state.DB_FILE = str(db_file)

    state = system_state.load_state()
    new_order = {'pos': (3,3,3), 'payload_kg': 3.0, 'id': 'Custom Order'}
    state['pending_orders']['Custom Order'] = new_order

    system_state.save_state(state)
    new_state = system_state.load_state()

    assert 'Custom Order' in new_state['pending_orders']
    assert new_state['pending_orders']['Custom Order']['payload_kg'] == 3.0