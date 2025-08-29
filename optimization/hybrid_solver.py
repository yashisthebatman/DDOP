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
        """Initializes the appropriate sampler based on config."""
        if config.USE_DWAVE_SAMPLER:
            try:
                # This requires your D-Wave API token to be configured
                return EmbeddingComposite(DWaveSampler())
            except Exception as e:
                print(f"WARN: Could not connect to DWaveSampler: {e}. Falling back to Neal.")
                return neal.SimulatedAnnealingSampler()
        else:
            print("INFO: Using local Simulated Annealing Sampler (Neal).")
            return neal.SimulatedAnnealingSampler()

    def solve_qubo(self, qubo):
        """Solves the formulated QUBO using a D-Wave sampler or a simulated annealer."""
        print("INFO: Solving QUBO with sampler...")
        sampleset = self.sampler.sample_qubo(qubo, num_reads=100)
        # Get the best solution (lowest energy)
        solution = sampleset.first.sample
        energy = sampleset.first.energy
        print(f"INFO: QUBO solver finished. Lowest energy: {energy:.2f}")
        return solution

    def solve_with_or_tools(self, time_matrix, energy_matrix, drones, orders, weights):
        """
        Solves the VRP using Google OR-Tools as a classical baseline.
        
        Args:
            time_matrix (np.ndarray): Matrix of predicted flight times between locations.
            energy_matrix (np.ndarray): Matrix of predicted energy consumption between locations.
            drones (list): List of Drone objects.
            orders (list): List of Order objects.
            weights (dict): A dictionary {'time': float, 'energy': float} defining optimization priority.
        """
        print("INFO: Solving with Google OR-Tools...")
        
        num_depots = len(drones)
        num_orders = len(orders)
        num_locations = num_depots + num_orders
        
        # Define start and end nodes for each vehicle (drone)
        depot_indices = [i for i in range(num_depots)]
        manager = pywrapcp.RoutingIndexManager(num_locations, num_depots, depot_indices, depot_indices)
        
        # Create Routing Model
        routing = pywrapcp.RoutingModel(manager)
        
        # --- KEY CHANGE: Use user-defined weights to create the final cost matrix ---
        combined_cost_matrix = (weights['time'] * time_matrix + weights['energy'] * energy_matrix)
        
        # Handle infinities (from no-fly zones) and convert to integers for OR-Tools
        if np.any(np.isfinite(combined_cost_matrix)):
             max_cost = np.max(combined_cost_matrix[np.isfinite(combined_cost_matrix)])
        else: # Handle case where all paths might be infinite
             max_cost = 1 
        combined_cost_matrix[np.isinf(combined_cost_matrix)] = int(max_cost * 100) # Use a large number for infinity
        int_cost_matrix = combined_cost_matrix.astype(int)

        def cost_callback(from_index, to_index):
            """Returns the cost between two nodes."""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int_cost_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(cost_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Set search parameters to find a good first solution quickly.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self._decode_or_tools_solution(solution, routing, manager, drones, orders)
        else:
            print("ERROR: OR-Tools found no solution.")
            return None

    def _decode_or_tools_solution(self, solution, routing, manager, drones, orders):
        """Decodes the OR-Tools solution into readable routes and node indices."""
        routes = {}
        for i in range(len(drones)):
            route_readable = []
            route_indices = []
            index = routing.Start(i)
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_indices.append(node_index)
                
                if node_index < len(drones):
                    route_readable.append(f"Depot-{node_index}")
                else:
                    # Adjust index to match the order list
                    order_id = orders[node_index - len(drones)].id
                    route_readable.append(f"Order-{order_id}")
                index = solution.Value(routing.NextVar(index))
            
            # Add the final depot to complete the loop
            end_node_index = manager.IndexToNode(index)
            route_indices.append(end_node_index)
            route_readable.append(f"Depot-{i}") # The end depot is the same as the start depot 'i'

            routes[drones[i].id] = {'readable': route_readable, 'indices': route_indices}
            
        return routes