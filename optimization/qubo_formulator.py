# optimization/qubo_formulator.py
import pyqubo
from pyqubo import Array
import numpy as np
import config

class QuboFormulator:
    """Formulates the drone delivery VRP as a QUBO."""
    # CHANGE: Added 'weights' as a parameter to the function
    def build_vrp_qubo(self, time_matrix, energy_matrix, drones, orders, weights):
        """Builds the QUBO model using pyqubo."""
        num_drones = len(drones)
        num_depots = num_drones
        num_orders = len(orders)
        num_locations = num_depots + num_orders
        
        x = Array.create('x', (num_drones, num_locations, num_locations), 'BINARY')
        
        # CHANGE: Use the passed-in 'weights' dictionary instead of config
        cost_matrix = weights['time'] * time_matrix + weights['energy'] * energy_matrix
        
        max_cost = np.max(cost_matrix[np.isfinite(cost_matrix)]) if np.any(np.isfinite(cost_matrix)) else 1
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
        num_orders = len(orders)
        num_locations = num_depots + num_orders
        
        for k in range(num_drones):
            route_indices, final_path = [-1] * num_locations, []
            for t in range(num_locations):
                for i in range(num_locations):
                    if sample.get(f'x[{k}][{t}][{i}]', 0) == 1:
                        route_indices[t] = i; break
            
            for loc_idx in route_indices:
                if loc_idx == -1: continue
                if not final_path or loc_idx != final_path[-1]:
                    final_path.append(loc_idx)

            readable_route = []
            for loc_idx in final_path:
                loc_name = loc_map.get(loc_idx)
                if loc_name: readable_route.append(loc_name)
            routes[k] = readable_route
        return routes