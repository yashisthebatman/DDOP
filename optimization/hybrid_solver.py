# optimization/hybrid_solver.py
import neal
from dwave.system import DWaveSampler, EmbeddingComposite
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

import config

class HybridSolver:
    """Solves the optimization problem using a hybrid of methods."""
    def __init__(self):
        self.sampler = self._get_sampler()

    def _get_sampler(self):
        if config.USE_DWAVE_SAMPLER:
            try:
                return EmbeddingComposite(DWaveSampler())
            except Exception as e:
                print(f"WARN: Could not connect to DWaveSampler: {e}. Falling back to Neal.")
                return neal.SimulatedAnnealingSampler()
        else:
            print("INFO: Using local Simulated Annealing Sampler (Neal).")
            return neal.SimulatedAnnealingSampler()

    def solve_qubo(self, qubo):
        """Solves the formulated QUBO."""
        print("INFO: Solving QUBO with sampler...")
        sampleset = self.sampler.sample_qubo(qubo, num_reads=100)
        solution = sampleset.first.sample
        energy = sampleset.first.energy
        print(f"INFO: QUBO solver finished. Lowest energy: {energy:.2f}")
        return solution

    def solve_with_or_tools(self, time_matrix, energy_matrix, drones, orders):
        """Solves the VRP using Google OR-Tools as a classical baseline."""
        print("INFO: Solving with Google OR-Tools...")
        
        num_depots = len(drones)
        num_orders = len(orders)
        num_locations = num_depots + num_orders
        
        # --- THE FIX IS HERE ---
        # For a multi-depot problem, the RoutingIndexManager requires two lists:
        # one for the start nodes of each vehicle, and one for the end nodes.
        depot_indices = [i for i in range(num_depots)]
        manager = pywrapcp.RoutingIndexManager(num_locations, num_depots, depot_indices, depot_indices)
        # --- END OF FIX ---

        routing = pywrapcp.RoutingModel(manager)
        
        combined_cost_matrix = (config.TIME_WEIGHT * time_matrix + config.ENERGY_WEIGHT * energy_matrix)
        max_cost = np.max(combined_cost_matrix[np.isfinite(combined_cost_matrix)])
        combined_cost_matrix[np.isinf(combined_cost_matrix)] = int(max_cost * 100)
        int_cost_matrix = combined_cost_matrix.astype(int)

        def cost_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int_cost_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(cost_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self._decode_or_tools_solution(solution, routing, manager, drones, orders)
        else:
            print("ERROR: OR-Tools found no solution.")
            return None

    def _decode_or_tools_solution(self, solution, routing, manager, drones, orders):
        """Decodes the OR-Tools solution into readable routes."""
        routes = {}
        for i in range(len(drones)):
            route = []
            route_indices = []
            index = routing.Start(i)
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_indices.append(node_index)
                
                if node_index < len(drones):
                    route.append(f"Depot-{node_index}")
                else:
                    order_id = orders[node_index - len(drones)].id
                    route.append(f"Order-{order_id}")
                index = solution.Value(routing.NextVar(index))
            
            end_node_index = manager.IndexToNode(index)
            route_indices.append(end_node_index)
            route.append(f"Depot-{i}")

            routes[drones[i].id] = {'readable': route, 'indices': route_indices}
        return routes