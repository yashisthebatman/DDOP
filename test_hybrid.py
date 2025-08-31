#!/usr/bin/env python3
"""
Test script for the hybrid A*+QUBO implementation
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import Environment, WeatherSystem
from ml_predictor.predictor import EnergyTimePredictor
from path_planner import PathPlanner3D
from config import WAYPOINTS

def test_hybrid_qubo_pathfinding():
    """Test the hybrid A*+QUBO pathfinding implementation."""
    
    print("=== Testing Hybrid A*+QUBO Path Planning ===")
    
    # Setup environment
    print("1. Setting up environment...")
    env = Environment(weather_system=WeatherSystem(max_speed=5.0))
    predictor = EnergyTimePredictor()
    planner = PathPlanner3D(env, predictor)
    
    # Test points
    start_pos = WAYPOINTS["Hub A"]  # (-74.013, 40.705, 50)
    end_pos = WAYPOINTS["WTC"]     # (-74.0134, 40.7127, 400.0)
    
    print(f"2. Testing path from {start_pos} to {end_pos}")
    
    # Test parameters
    payload_kg = 2.0
    weights = {'time': 0.5, 'energy': 0.5}
    
    # Test hybrid A*+QUBO approach
    print("3. Testing Hybrid A*+QUBO approach...")
    try:
        hybrid_path, hybrid_status = planner.find_hybrid_qubo_path(
            start_pos, end_pos, payload_kg, weights, use_qubo=True
        )
        
        if hybrid_path:
            print(f"   ✅ Hybrid path found: {len(hybrid_path)} waypoints")
            print(f"   Status: {hybrid_status}")
            cost = planner._calculate_path_cost(hybrid_path, payload_kg, weights)
            print(f"   Cost: {cost:.2f}")
        else:
            print(f"   ❌ Hybrid path failed: {hybrid_status}")
            
    except Exception as e:
        print(f"   ⚠️  Hybrid approach failed with error: {e}")
        print("   This is expected if dependencies are missing")
    
    # Test fallback A* approach  
    print("4. Testing A* fallback approach...")
    try:
        astar_path, astar_status = planner.find_hybrid_qubo_path(
            start_pos, end_pos, payload_kg, weights, use_qubo=False
        )
        
        if astar_path:
            print(f"   ✅ A* path found: {len(astar_path)} waypoints")
            print(f"   Status: {astar_status}")
            cost = planner._calculate_path_cost(astar_path, payload_kg, weights)
            print(f"   Cost: {cost:.2f}")
        else:
            print(f"   ❌ A* path failed: {astar_status}")
            
    except Exception as e:
        print(f"   ❌ A* approach failed: {e}")
    
    # Test baseline approach
    print("5. Testing baseline A* approach...")
    try:
        baseline_path = planner.find_baseline_path(start_pos, end_pos, payload_kg, weights)
        
        if baseline_path:
            print(f"   ✅ Baseline path found: {len(baseline_path)} waypoints")
            cost = planner._calculate_path_cost(baseline_path, payload_kg, weights)
            print(f"   Cost: {cost:.2f}")
        else:
            print("   ❌ Baseline path failed")
            
    except Exception as e:
        print(f"   ❌ Baseline approach failed: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_hybrid_qubo_pathfinding()