import pytest
import numpy as np
from ml_predictor.predictor import PhysicsBasedPredictor
# FIX: Import constants from config to use in tests, fixing the AttributeError.
from config import DRONE_SPEED_MPS, DRONE_VERTICAL_SPEED_MPS

@pytest.fixture
def predictor():
    """Provides a fresh PhysicsBasedPredictor instance."""
    return PhysicsBasedPredictor()

def test_zero_distance(predictor):
    """Test that moving zero distance costs zero time and energy."""
    p1 = (0, 0, 100)
    time, energy = predictor.predict(p1, p1, payload_kg=1.0, wind_vector=np.array([0,0,0]))
    assert time == 0
    assert energy == 0

def test_horizontal_flight(predictor):
    """Test basic horizontal flight costs."""
    p1 = (0, 0, 100)
    p2 = (1000, 0, 100) # 1km flight
    time, energy = predictor.predict(p1, p2, payload_kg=1.0, wind_vector=np.array([0,0,0]))
    
    # FIX: Use the imported constant for the assertion.
    assert time == pytest.approx(1000 / DRONE_SPEED_MPS)
    assert energy > 0

def test_vertical_flight(predictor):
    """Test that climbing costs more time and energy than descending."""
    p1 = (0, 0, 100)
    p_up = (0, 0, 200)
    p_down = (0, 0, 0)
    
    time_up, energy_up = predictor.predict(p1, p_up, 1.0, np.array([0,0,0]))
    time_down, energy_down = predictor.predict(p1, p_down, 1.0, np.array([0,0,0]))

    # FIX: Use the imported constant for the assertion.
    assert time_up == pytest.approx(100 / DRONE_VERTICAL_SPEED_MPS)
    assert energy_up > energy_down
    assert energy_down > 0

def test_payload_impact(predictor):
    """Test that a heavier payload increases energy consumption."""
    p1 = (0, 0, 100)
    p2 = (0, 0, 200) # Climbing
    
    _, energy_light = predictor.predict(p1, p2, 0.5, np.array([0,0,0]))
    _, energy_heavy = predictor.predict(p1, p2, 5.0, np.array([0,0,0]))

    assert energy_heavy > energy_light

def test_wind_impact(predictor):
    """Test that headwinds increase cost and tailwinds decrease it."""
    p1 = (0, 0, 100)
    p2 = (1000, 0, 100)
    
    headwind = np.array([-5, 0, 0])
    time_head, energy_head = predictor.predict(p1, p2, 1.0, headwind)

    tailwind = np.array([5, 0, 0])
    time_tail, energy_tail = predictor.predict(p1, p2, 1.0, tailwind)

    time_no, energy_no = predictor.predict(p1, p2, 1.0, np.array([0,0,0]))
    
    assert time_head > time_no > time_tail
    assert energy_head > energy_no > energy_tail

def test_turning_energy(predictor):
    """Test that turning costs a small amount of energy."""
    p_prev, p1 = (-100, 0, 100), (0, 0, 100)
    p2_straight, p2_turn = (100, 0, 100), (0, 100, 100)
    
    _, energy_straight = predictor.predict(p1, p2_straight, 1.0, np.array([0,0,0]), p_prev)
    _, energy_turn = predictor.predict(p1, p2_turn, 1.0, np.array([0,0,0]), p_prev)
    
    assert energy_turn > energy_straight