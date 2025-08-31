# ==============================================================================
# generate_heuristic.py
# ==============================================================================
import multiprocessing
import itertools
import pickle
import time
import numpy as np

# --- Assume these are imported from your project structure ---
from path_planner import PathPlanner3D
from environment import Environment, WeatherSystem
from ml_predictor.predictor import EnergyTimePredictor
from config import WAYPOINTS, DEFAULT_CRUISING_ALTITUDE
from utils.heuristics import a_star_search # Import a_star directly
# -----------------------------------------------------------

# --- NEW: Global variables to be inherited by worker processes ---
main_planner_grid = None
main_planner_moves = None

def init_worker(grid, moves):
    """Initializer now just receives the grid and moves, no complex objects."""
    global main_planner_grid, main_planner_moves
    main_planner_grid = grid
    main_planner_moves = moves
    print(f"[{multiprocessing.current_process().name}] Worker initialized.")

def compute_path_for_pair_simple(args):
    """
    A much simpler worker. It doesn't know about planners, only grids.
    It receives grid coordinates and returns grid coordinates.
    """
    start_name, end_name, start_coord, end_coord = args
    global main_planner_grid, main_planner_moves
    
    print(f"[{multiprocessing.current_process().name}] Computing path: {start_name} -> {end_name}...")
    
    # Pathfind directly on the grid coordinates
    path_coords = a_star_search(start_coord, end_coord, main_planner_grid, main_planner_moves)
    
    if path_coords:
        return (start_name, end_name, path_coords)

    print(f"  -> FAILED to find path for {start_name} -> {end_name}")
    return (start_name, end_name, None)

if __name__ == "__main__":
    start_time = time.time()
    
    print("--- Step 1: Creating a single authoritative planner ---")
    
    # Create ONE planner that will be the source of all truth
    authoritative_env = Environment(weather_system=WeatherSystem(max_speed=0))
    authoritative_planner = PathPlanner3D(authoritative_env, EnergyTimePredictor())
    
    print("\n--- Step 2: Pre-validating waypoints and detecting collisions ---")
    
    snapped_coords = {}
    for name, pos_orig in WAYPOINTS.items():
        z = pos_orig[2] if pos_orig[2] > DEFAULT_CRUISING_ALTITUDE else DEFAULT_CRUISING_ALTITUDE
        cruising_pos = (pos_orig[0], pos_orig[1], z)
        
        grid_coord = authoritative_planner._world_to_grid(cruising_pos)
        valid_grid_coord = authoritative_planner._find_nearest_valid_node(grid_coord)
        
        if valid_grid_coord:
            print(f"  Waypoint '{name}' snaps to grid coordinate {valid_grid_coord}")
            snapped_coords[name] = valid_grid_coord
        else:
            print(f"  -> FATAL ERROR: Could not find a valid node for waypoint '{name}'. It will be excluded.")

    valid_waypoints_coords = {}
    coord_to_name_map = {}
    for name, coord in snapped_coords.items():
        if coord in coord_to_name_map:
            existing_name = coord_to_name_map[coord]
            print(f"  -> COLLISION DETECTED: Waypoint '{name}' at {coord} collides with '{existing_name}'. '{name}' will be excluded.")
        else:
            coord_to_name_map[coord] = name
            valid_waypoints_coords[name] = coord

    print(f"\nValidation complete. Using {len(valid_waypoints_coords)} unique, valid waypoints for heuristic generation.")
    
    final_waypoint_names = list(valid_waypoints_coords.keys())
    
    # Create tasks using the GRID coordinates from our single planner
    tasks = [
        (p[0], p[1], valid_waypoints_coords[p[0]], valid_waypoints_coords[p[1]]) 
        for p in itertools.permutations(final_waypoint_names, 2)
    ]

    print(f"\n--- Step 3: Starting heuristic generation for {len(tasks)} waypoint pairs ---")
    print(f"Using {multiprocessing.cpu_count()} CPU cores...\n")
    
    # Initialize the pool with the grid and moves from our single planner
    pool_args = (authoritative_planner.grid, authoritative_planner.moves)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=init_worker, initargs=pool_args) as pool:
        results = pool.map(compute_path_for_pair_simple, tasks)

    print("\n--- Step 4: Calculating costs and building heuristic table ---")
    
    heuristic_table = {name: {} for name in final_waypoint_names}
    successful_paths = 0
    for start, end, path_grid_coords in results:
        if path_grid_coords:
            # Convert the grid path back to a world path
            world_path = [authoritative_planner._grid_to_world(c) for c in path_grid_coords]
            
            # Calculate cost using the single authoritative planner
            weights = {'time': 0.5, 'energy': 0.5}
            cost = authoritative_planner._calculate_path_cost(world_path, 0, weights, use_zero_wind=True)
            
            if cost > 0:
                print(f"  -> Successfully costed path {start} -> {end} with cost {cost:.2f}")
                heuristic_table[start][end] = (cost, world_path)
                successful_paths += 1
            else:
                 print(f"  -> WARNING: Post-calculation resulted in zero cost for {start} -> {end}. Discarding.")

    file_name = "quantum_heuristic.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(heuristic_table, f)
        
    end_time = time.time()
    
    print(f"\n--- Generation Complete ---")
    print(f"Successfully calculated and costed {successful_paths} out of {len(tasks)} paths.")
    print(f"✅ Quantum heuristic table generated and saved to '{file_name}' in {end_time - start_time:.2f} seconds.")