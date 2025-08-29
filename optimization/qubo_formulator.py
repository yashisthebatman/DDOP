# optimization/qubo_formulator.py
import pyqubo
from pyqubo import Array
import numpy as np

import config

class QuboFormulator:
    """Formulates the drone delivery VRP as a QUBO."""
    def build_vrp_qubo(self, time_matrix, energy_matrix, drones, orders):
        """Builds the QUBO model using pyqubo."""
        num_drones = len(drones)
        num_depots = num_drones
        num_orders = len(orders)
        num_locations = num_depots + num_orders
        
        x = Array.create('x', (num_drones, num_locations, num_locations), 'BINARY')
        
        cost_matrix = config.TIME_WEIGHT * time_matrix + config.ENERGY_WEIGHT * energy_matrix
        # Replace inf with a large number for QUBO formulation
        max_cost = np.max(cost_matrix[np.isfinite(cost_matrix)])
        cost_matrix[np.isinf(cost_matrix)] = max_cost * 100

        objective_func = 0
        for k in range(num_drones):
            for t in range(num_locations - 1):
                for i in range(num_locations):
                    for j in range(num_locations):
                        objective_func += cost_matrix[i, j] * x[k, t, i] * x[k, t + 1, j]
        
        constraint_visit_orders = 0
        for j in range(num_depots, num_locations):
            term = sum(x[k, t, j] for k in range(num_drones) for t in range(num_locations))
            constraint_visit_orders += (term - 1)**2
            
        constraint_one_loc_per_drone = 0
        for k in range(num_drones):
            for t in range(num_locations):
                term = sum(x[k, t, i] for i in range(num_locations))
                constraint_one_loc_per_drone += (term - 1)**2

        constraint_start_depot = 0
        for k in range(num_drones):
            constraint_start_depot += (x[k, 0, k] - 1)**2

        H = objective_func + \
            config.PENALTY_ORDER_NOT_VISITED * constraint_visit_orders + \
            config.PENALTY_ONE_DRONE_PER_TIMESTEP * constraint_one_loc_per_drone + \
            config.PENALTY_DRONE_ROUTE_CONTINUITY * constraint_start_depot
            
        model = H.compile()
        return model.to_qubo()

    def decode_solution(self, sample, drones, orders, loc_map):
        """Decodes the binary solution from the solver into routes."""
        routes = {d.id: [] for d in drones}
        num_drones = len(drones)
        num_depots = num_drones
        num_locations = num_depots + len(orders)
        
        for k in range(num_drones):
            # Reconstruct the ordered sequence of location indices for drone k
            route_indices = [-1] * num_locations
            for t in range(num_locations):
                for i in range(num_locations):
                    if sample.get(f'x[{k}][{t}][{i}]', 0) == 1:
                        route_indices[t] = i
                        break
            
            # Convert indices to readable names and remove duplicates/unused steps
            final_path = []
            for loc_idx in route_indices:
                if loc_idx == -1: continue # Skip unassigned time steps
                
                # Add location to path only if it's new
                if not final_path or loc_idx != final_path[-1]:
                    final_path.append(loc_idx)

            # Convert location indices to human-readable names
            readable_route = []
            for loc_idx in final_path:
                loc_info = loc_map[loc_idx]
                if loc_info['type'] == 'depot':
                    readable_route.append(f"Depot-{loc_info['obj'].id}")
                elif loc_info['type'] == 'order':
                    readable_route.append(f"Order-{loc_info['obj'].id}")
            routes[k] = readable_route
        return routes