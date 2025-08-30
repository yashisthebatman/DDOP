# generate_heuristic.py
import multiprocessing
import itertools
import pickle
import time
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from path_planner import PathPlanner3D
from environment import Environment, WeatherSystem
from ml_predictor.predictor import EnergyTimePredictor
import config

def compute_path_for_pair(args):
    """
    Worker function now receives pre-validated start and end coordinates.
    """
    start_name, end_name, start_pos, end_pos = args
    
    # Each process gets its own clean environment
    zero_wind_weather = WeatherSystem(max_speed=0)
    env = Environment(weather_system=zero_wind_weather)
    pred = EnergyTimePredictor()
    planner = PathPlanner3D(env, pred)

    print(f"[{multiprocessing.current_process().name}] Computing path: {start_name} -> {end_name}...")
    
    payload_kg = 0
    weights = {'time': 0.5, 'energy': 0.5}
    path = planner.solve_path_qubo(start_pos, end_pos, payload_kg, weights)

    if path:
        cost = planner._calculate_path_cost(path, payload_kg, weights, use_zero_wind=True)
        print(f"  -> Path found for {start_name} -> {end_name} with baseline cost {cost:.2f}")
        return (start_name, end_name, (cost, path))
        
    print(f"  -> FAILED to find path for {start_name} -> {end_name}")
    return (start_name, end_name, None)

if __name__ == "__main__":
    start_time = time.time()
    
    # --- THE ULTIMATE FIX: PRE-VALIDATION AT CRUISING ALTITUDE ---
    print("Pre-validating waypoints against the 3D environment grid...")
    
    # Create a single planner instance to access its grid and validation tools
    temp_env = Environment(weather_system=WeatherSystem(max_speed=0))
    validation_planner = PathPlanner3D(temp_env, EnergyTimePredictor())
    
    valid_waypoints = {}
    
    for name, pos_orig in config.WAYPOINTS.items():
        # 1. ALWAYS determine the cruising altitude for this strategic waypoint
        z = pos_orig[2] if pos_orig[2] > config.DEFAULT_CRUISING_ALTITUDE else config.DEFAULT_CRUISING_ALTITUDE
        cruising_pos = (pos_orig[0], pos_orig[1], z)
        
        # 2. Convert to a grid coordinate
        grid_coord = validation_planner._world_to_grid(cruising_pos)
        
        # 3. Snap to the nearest valid, open-air grid cell
        valid_grid_coord = validation_planner._find_nearest_valid_node(grid_coord)
        
        if valid_grid_coord:
            # 4. CRITICAL: Check for duplicates after snapping.
            if valid_grid_coord not in valid_waypoints.values():
                valid_waypoints[name] = valid_grid_coord
            else:
                # This waypoint is too close to another one and causes a collision.
                print(f"  -> WARNING: Waypoint '{name}' snaps to a duplicate grid location and will be excluded.")
        else:
            print(f"  -> ERROR: Could not find a valid node for waypoint '{name}' and it will be excluded.")

    print(f"Validation complete. Using {len(valid_waypoints)} unique, valid waypoints for heuristic generation.")
    
    # Use the world positions of these unique, validated grid points for the pathfinder
    validated_world_positions = {name: validation_planner._grid_to_world(coord) for name, coord in valid_waypoints.items()}
    
    waypoint_names = list(validated_world_positions.keys())
    tasks = [(p[0], p[1], validated_world_positions[p[0]], validated_world_positions[p[1]]) for p in itertools.permutations(waypoint_names, 2)]

    print(f"Starting heuristic generation for {len(tasks)} waypoint pairs using {multiprocessing.cpu_count()} CPU cores.")
    
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = pool.map(compute_path_for_pair, tasks)

    heuristic_table = {name: {} for name in waypoint_names}
    for start, end, data in results:
        if data:
            heuristic_table[start][end] = data

    file_name = "quantum_heuristic.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(heuristic_table, f)
        
    end_time = time.time()
    print(f"\n✅ Quantum heuristic table generated and saved to '{file_name}' in {end_time - start_time:.2f} seconds.")