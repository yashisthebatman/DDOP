# main.py

# --- FIX IS HERE: Add project root to Python's path ---
import sys
import os
# This ensures that the script can find other modules in the same directory (like environment.py)
# and sub-directories (like utils/ and optimization/).
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# --- END OF FIX ---

from environment import Environment
from ml_predictor.predictor import EnergyTimePredictor
from optimization.qubo_formulator import QuboFormulator
from optimization.hybrid_solver import HybridSolver
from utils.reporting import print_solution_summary
import config
import numpy as np

def main():
    """Main orchestration script for the Q-DOP project."""
    print("--- Quantum-Hybrid, ML-Enhanced Drone Optimization System (Q-DOP) ---")

    env = Environment(seed=42)
    predictor = EnergyTimePredictor()
    qubo_formulator = QuboFormulator()
    solver = HybridSolver()
    
    print(f"\nScenario: {len(env.drones)} drones, {len(env.orders)} orders.")
    print(f"Wind Vector: {env.wind_vector[:2]} m/s")
    print(f"No-Fly Zones: {len(env.no_fly_zones)}")

    locations = [d.start_location for d in env.drones] + [o.location for o in env.orders]
    
    avg_payload = np.mean([o.payload_kg for o in env.orders]) if env.orders else 0
    time_matrix, energy_matrix = predictor.build_cost_matrices(locations, env, avg_payload)
    
    # --- Solve with QUBO (Quantum or Simulated Annealing) ---
    qubo, offset = qubo_formulator.build_vrp_qubo(time_matrix, energy_matrix, env.drones, env.orders)
    print(f"\nINFO: QUBO generated with {len(qubo)} variables.")
    
    qubo_solution_sample = solver.solve_qubo(qubo)
    
    qubo_loc_map = {}
    idx=0
    for d in env.drones: qubo_loc_map[idx] = {'type': 'depot', 'obj': d}; idx+=1
    for o in env.orders: qubo_loc_map[idx] = {'type': 'order', 'obj': o}; idx+=1

    decoded_qubo_routes = qubo_formulator.decode_solution(qubo_solution_sample, env.drones, env.orders, qubo_loc_map)
    print("\n--- QUBO-based Solution ---")
    for drone_id, readable_route in decoded_qubo_routes.items():
        if len(readable_route) > 1 and readable_route[0] == readable_route[-1]:
             print(f"  Drone {drone_id}: Idle")
        else:
            print(f"  Drone {drone_id}: {' -> '.join(readable_route)}")
    
    # --- Solve with Google OR-Tools (Classical Baseline) ---
    if config.USE_OR_TOOLS_SOLVER:
        or_tools_routes = solver.solve_with_or_tools(time_matrix, energy_matrix, env.drones, env.orders)
        
        if or_tools_routes:
            print_solution_summary("Classical OR-Tools", or_tools_routes, time_matrix, energy_matrix)
    
    print("\n--- Project Execution Finished ---")

if __name__ == "__main__":
    main()