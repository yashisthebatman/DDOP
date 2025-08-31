import multiprocessing
import itertools
import pickle
import time
import numpy as np
from functools import partial

# --- Assume these are imported from your project structure ---
from path_planner import PathPlanner3D
from environment import Environment, WeatherSystem
from ml_predictor.predictor import EnergyTimePredictor
from config import WAYPOINTS
from utils.heuristics import a_star_search
from qubo_solver import solve_tsp_qubo
# -----------------------------------------------------------

# Global planner for worker processes to prevent re-initialization
_planner = None

def init_worker(planner_instance):
    """Initializes a worker process with a shared planner instance."""
    global _planner
    _planner = planner_instance
    print(f"[{multiprocessing.current_process().name}] Worker initialized.")

def compute_baseline_path(pair):
    """Worker function to compute the A* path and cost between two waypoints."""
    global _planner
    start_name, end_name = pair
    start_pos, end_pos = WAYPOINTS[start_name], WAYPOINTS[end_name]
    
    # Use the planner's internal baseline pathfinding (no dynamic conditions)
    world_path = _planner.find_baseline_path(start_pos, end_pos)
    
    if not world_path:
        print(f"  -> FAILED to find A* path for {start_name} -> {end_name}")
        return start_name, end_name, None, float('inf')

    # Calculate cost using zero wind to establish a stable baseline
    cost = _planner._calculate_path_cost(world_path, payload_kg=0, weights={'time': 0.5, 'energy': 0.5}, use_zero_wind=True)
    
    print(f"  -> A* path computed for {start_name} -> {end_name} with cost {cost:.2f}")
    return start_name, end_name, world_path, cost

if __name__ == "__main__":
    start_time = time.time()

    print("--- Step 1: Initializing Authoritative Planner ---")
    # This single planner is the source of truth for the grid and physics
    authoritative_env = Environment(weather_system=WeatherSystem(max_speed=0))
    authoritative_planner = PathPlanner3D(authoritative_env, EnergyTimePredictor())
    waypoint_names = list(WAYPOINTS.keys())
    
    print(f"\n--- Step 2: Computing All-Pairs Baseline A* Paths ({len(waypoint_names)**2} pairs) ---")
    # Create all possible pairs of waypoints to calculate direct paths
    tasks = list(itertools.product(waypoint_names, repeat=2))
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=init_worker, initargs=(authoritative_planner,)) as pool:
        results = pool.map(compute_baseline_path, tasks)

    # Store baseline paths and create the cost matrix for the QUBO solver
    baseline_paths = {}
    cost_matrix = {name: {other: float('inf') for other in waypoint_names} for name in waypoint_names}
    for start, end, path, cost in results:
        cost_matrix[start][end] = cost
        if path:
            baseline_paths[(start, end)] = path

    print("\n--- Step 3: Generating Strategic Routes with QUBO Solver ---")
    # For every possible mission (start -> end), find the optimal sequence of waypoints
    strategic_routes = {name: {} for name in waypoint_names}
    
    missions = list(itertools.permutations(waypoint_names, 2))
    for start_node, end_node in missions:
        print(f"  Solving QUBO for mission: {start_node} -> {end_node}")
        
        # The QUBO solver finds the best order to visit intermediate nodes
        # It's a TSP problem where start and end points are fixed
        optimal_sequence = solve_tsp_qubo(cost_matrix, waypoint_names, start_node, end_node)
        
        if not optimal_sequence:
            print(f"  -> QUBO FAILED for {start_node} -> {end_node}. Skipping.")
            continue
            
        # Stitch together the A* paths for the sequence returned by QUBO
        stitched_path = []
        total_cost = 0
        for i in range(len(optimal_sequence) - 1):
            leg_start, leg_end = optimal_sequence[i], optimal_sequence[i+1]
            path_segment = baseline_paths.get((leg_start, leg_end))
            if path_segment:
                # Avoid duplicating the connection point
                stitched_path.extend(path_segment if i == 0 else path_segment[1:])
                total_cost += cost_matrix[leg_start][leg_end]
            else:
                print(f"  -> CRITICAL ERROR: Missing baseline path for {leg_start}->{leg_end}. Aborting mission stitch.")
                stitched_path = None
                break
        
        if stitched_path:
            print(f"  -> Strategic route found for {start_node} -> {end_node}: {' -> '.join(optimal_sequence)} with cost {total_cost:.2f}")
            strategic_routes[start_node][end_node] = {
                "cost": total_cost,
                "sequence": optimal_sequence,
                "path": stitched_path
            }

    file_name = "quantum_heuristic.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(strategic_routes, f)
        
    end_time = time.time()
    
    print(f"\n--- Generation Complete ---")
    print(f"Successfully generated {len(missions)} strategic routes.")
    print(f"✅ Hybrid QUBO-A* heuristic table saved to '{file_name}' in {end_time - start_time:.2f} seconds.")
    print("\nIMPORTANT: After changing grid resolution or waypoints, you MUST re-run this script.")