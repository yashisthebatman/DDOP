# utils/reporting.py
import numpy as np

def calculate_route_cost(route_indices, time_matrix, energy_matrix):
    """Calculates the total time and energy for a given route of indices."""
    total_time = 0
    total_energy = 0
    for i in range(len(route_indices) - 1):
        u = route_indices[i]
        v = route_indices[i+1]
        
        time = time_matrix[u, v]
        energy = energy_matrix[u, v]

        if np.isinf(time) or np.isinf(energy):
            print(f"WARN: Invalid segment detected in route: {u}->{v}")
            return np.inf, np.inf

        total_time += time
        total_energy += energy
        
    return total_time, total_energy

def print_solution_summary(title, routes, time_matrix, energy_matrix):
    """Prints a formatted summary of the solution."""
    print(f"\n--- {title} Solution ---")
    grand_total_time = 0
    grand_total_energy = 0

    if not routes:
        print("  No routes found.")
        return

    for drone_id, route_data in routes.items():
        readable_route = route_data['readable']
        route_indices = route_data['indices']

        if len(readable_route) <= 2: # e.g., Depot-0 -> Depot-0
            print(f"  Drone {drone_id}: Idle")
            continue
        
        time, energy = calculate_route_cost(route_indices, time_matrix, energy_matrix)
        grand_total_time += time
        grand_total_energy += energy
        
        print(f"  Drone {drone_id}: {' -> '.join(readable_route)}")
        print(f"    - Route Time: {time:.2f}s ({time/60:.2f} min)")
        print(f"    - Route Energy: {energy:.2f} Wh")

    print("-" * 25)
    print(f"  Total Mission Time (Sum of all routes): {grand_total_time:.2f}s")
    print(f"  Total Mission Energy: {grand_total_energy:.2f} Wh")
    print("-" * 25)