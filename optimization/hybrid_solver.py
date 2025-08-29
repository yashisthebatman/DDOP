# optimization/hybrid_solver.py
import neal
from dwave.system import DWaveSampler, EmbeddingComposite
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import config

class HybridSolver:
    def __init__(self):
        self.sampler = self._get_sampler()

    def _get_sampler(self):
        if config.USE_DWAVE_SAMPLER:
            try: return EmbeddingComposite(DWaveSampler())
            except Exception as e:
                print(f"WARN: Could not connect to DWaveSampler: {e}. Falling back to Neal.")
                return neal.SimulatedAnnealingSampler()
        else:
            print("INFO: Using local Simulated Annealing Sampler (Neal).")
            return neal.SimulatedAnnealingSampler()

    def solve_qubo(self, qubo):
        print("INFO: Solving QUBO with sampler...")
        sampleset = self.sampler.sample_qubo(qubo, num_reads=100)
        solution, energy = sampleset.first.sample, sampleset.first.energy
        print(f"INFO: QUBO solver finished. Lowest energy: {energy:.2f}")
        return solution

    def solve_with_or_tools(self, time_matrix, energy_matrix, drones, orders):
        print("INFO: Solving with Google OR-Tools...")
        num_depots, num_orders = len(drones), len(orders)
        num_locations = num_depots + num_orders
        
        depot_indices = [i for i in range(num_depots)]
        manager = pywrapcp.RoutingIndexManager(num_locations, num_depots, depot_indices, depot_indices)
        routing = pywrapcp.RoutingModel(manager)
        
        combined_cost_matrix = (config.TIME_WEIGHT * time_matrix + config.ENERGY_WEIGHT * energy_matrix)
        max_cost = np.max(combined_cost_matrix[np.isfinite(combined_cost_matrix)])
        combined_cost_matrix[np.isinf(combined_cost_matrix)] = int(max_cost * 100)
        int_cost_matrix = combined_cost_matrix.astype(int)

        def cost_callback(from_index, to_index):
            from_node, to_node = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
            return int_cost_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(cost_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution: return self._decode_or_tools_solution(solution, routing, manager, drones, orders)
        else: print("ERROR: OR-Tools found no solution."); return None

    def _decode_or_tools_solution(self, solution, routing, manager, drones, orders):
        routes = {}
        for i in range(len(drones)):
            route, route_indices = [], []
            index = routing.Start(i)
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_indices.append(node_index)
                if node_index < len(drones): route.append(f"Depot-{node_index}")
                else: route.append(f"Order-{orders[node_index - len(drones)].id}")
                index = solution.Value(routing.NextVar(index))
            
            route_indices.append(manager.IndexToNode(index))
            route.append(f"Depot-{i}")
            routes[drones[i].id] = {'readable': route, 'indices': route_indices}
        return routes