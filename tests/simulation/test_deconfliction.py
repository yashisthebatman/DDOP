# FILE: tests/simulation/test_deconfliction.py

import pytest
from simulation.deconfliction import check_and_resolve_conflicts
from utils.coordinate_manager import CoordinateManager
from config import AVOIDANCE_MANEUVER_ALTITUDE_SEP

@pytest.fixture
def mock_drones():
    """Provides a dictionary of two drones for testing."""
    return {
        "Drone-A": {
            "id": "Drone-A",
            "pos": (-74.0, 40.72, 100.0),
            "status": "EN ROUTE"
        },
        "Drone-B": {
            "id": "Drone-B",
            "pos": (-74.0, 40.72, 100.0), # Initially at the same spot
            "status": "EN ROUTE"
        }
    }

def test_conflict_is_detected(mock_drones):
    """Manually place two drones within the safety bubble and assert they start avoiding."""
    coord_manager = CoordinateManager()
    drones = mock_drones
    
    # Place Drone B very close to Drone A (10 meters away horizontally)
    drones["Drone-B"]["pos"] = (-74.0001, 40.72, 100.0)

    check_and_resolve_conflicts(drones, coord_manager)
    
    assert drones["Drone-A"]["status"] == "AVOIDING"
    assert "original_status_before_avoid" in drones["Drone-A"]
    assert drones["Drone-B"]["status"] == "AVOIDING"
    assert "original_status_before_avoid" in drones["Drone-B"]

def test_avoidance_maneuver_sets_correct_targets(mock_drones):
    """After triggering a conflict, assert target altitudes are correct."""
    coord_manager = CoordinateManager()
    drones = mock_drones
    initial_alt = drones["Drone-A"]["pos"][2]

    # Place drones close enough to trigger conflict
    drones["Drone-B"]["pos"] = (-74.0001, 40.72, 100.0)

    check_and_resolve_conflicts(drones, coord_manager)
    
    # Drone-A is alphabetically smaller, so it should climb
    drone_a_target_alt = drones["Drone-A"]["avoidance_target_pos"][2]
    assert drone_a_target_alt == pytest.approx(initial_alt + AVOIDANCE_MANEUVER_ALTITUDE_SEP)
    
    # Drone-B should descend
    drone_b_target_alt = drones["Drone-B"]["avoidance_target_pos"][2]
    assert drone_b_target_alt == pytest.approx(initial_alt - AVOIDANCE_MANEUVER_ALTITUDE_SEP)