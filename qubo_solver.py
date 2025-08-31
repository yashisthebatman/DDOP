
import numpy as np
from pyqubo import Array, Constraint, solve_qubo
from dwave.samplers import SimulatedAnnealingSampler

def solve_tsp_qubo(cost_matrix, node_names, start_node, end_node):
    """
    Solves the Traveling Salesperson Problem (TSP) with fixed start and end points using QUBO.
    This function finds the optimal sequence of intermediate nodes to visit.

    Args:
        cost_matrix (dict): A nested dictionary where cost_matrix[i][j] is the cost from node i to j.
        node_names (list): A list of all possible node names.
        start_node (str): The name of the starting node.
        end_node (str): The name of the ending node.

    Returns:
        list: The optimal sequence of nodes from start to end, or None if no solution is found.
    """
    
    # Identify intermediate nodes that can be visited
    intermediate_nodes = [n for n in node_names if n not in {start_node, end_node}]
    num_intermediate = len(intermediate_nodes)
    
    # If there are no intermediate nodes, the path is direct
    if num_intermediate == 0:
        return [start_node, end_node]

    # Create a mapping from node names to indices for easier matrix access
    node_to_idx = {name: i for i, name in enumerate(node_names)}
    idx_to_node = {i: name for i, name in enumerate(node_names)}
    
    # Convert cost dictionary to a numpy array
    num_nodes = len(node_names)
    costs = np.full((num_nodes, num_nodes), 1e6) # Use a large number for non-existent paths
    for i in node_names:
        for j in node_names:
            if i in cost_matrix and j in cost_matrix[i]:
                costs[node_to_idx[i], node_to_idx[j]] = cost_matrix[i][j]

    # --- QUBO Formulation ---
    # x_i,p is a binary variable: 1 if intermediate node i is at position p in the path
    x = Array.create('x', shape=(num_intermediate, num_intermediate), vartype='BINARY')
    
    # --- Objective Function: Minimize total travel distance ---
    # Part 1: Cost from start_node to the first intermediate node
    cost_from_start = sum(costs[node_to_idx[start_node], node_to_idx[intermediate_nodes[i]]] * x[i, 0] for i in range(num_intermediate))
    
    # Part 2: Cost between intermediate nodes
    cost_between_intermediate = sum(
        costs[node_to_idx[intermediate_nodes[i]], node_to_idx[intermediate_nodes[j]]] * x[i, p] * x[j, p + 1]
        for i in range(num_intermediate) for j in range(num_intermediate) for p in range(num_intermediate - 1)
    )
    
    # Part 3: Cost from the last intermediate node to the end_node
    cost_to_end = sum(costs[node_to_idx[intermediate_nodes[i]], node_to_idx[end_node]] * x[i, num_intermediate - 1] for i in range(num_intermediate))
    
    objective = cost_from_start + cost_between_intermediate + cost_to_end

    # --- Constraints ---
    # Constraint 1: Each intermediate node must be visited exactly once.
    constraint_nodes = sum((sum(x[i, p] for p in range(num_intermediate)) - 1)**2 for i in range(num_intermediate))
    
    # Constraint 2: Each position in the path must be filled exactly once.
    constraint_positions = sum((sum(x[i, p] for i in range(num_intermediate)) - 1)**2 for p in range(num_intermediate))
    
    # Penalty term should be larger than the maximum possible cost to enforce constraints
    penalty = np.max(costs[costs < 1e5]) * (num_nodes**2) if np.any(costs < 1e5) else 1000

    # Combine into a single Hamiltonian
    H = objective + penalty * (constraint_nodes + constraint_positions)

    # --- Solve the QUBO ---
    model = H.compile()
    qubo, offset = model.to_qubo()
    
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(qubo, num_reads=100, chain_strength=penalty)
    
    # Decode the result
    decoded_sample = model.decode_sample(sampleset.first.sample, vartype='BINARY')
    
    if not decoded_sample.constraints(only_broken=True):
        path = [0] * num_intermediate
        for i in range(num_intermediate):
            for p in range(num_intermediate):
                if decoded_sample.array('x', (i, p)) == 1:
                    path[p] = intermediate_nodes[i]
        
        return [start_node] + path + [end_node]
    else:
        print(f"  -> WARNING: QUBO solver for {start_node}->{end_node} failed to find a valid solution.")
        # Fallback to direct path if QUBO fails
        return [start_node, end_node]

# --- Notes on Extending to a Multi-Drone Vehicle Routing Problem (VRP) ---
# To solve a VRP, the QUBO formulation would need to be expanded:
#
# 1. Binary Variables:
#    - You would need a variable like x_i,p,k which is 1 if order i is at position p in the route for drone k.
#
# 2. Objective Function:
#    - The objective would be to minimize the sum of costs over all drones 'k'.
#
# 3. Constraints:
#    - Each order 'i' must be delivered exactly once across all drones and all positions.
#    - Each position 'p' for each drone 'k' can be filled at most once.
#    - Capacity Constraint: The sum of payload_kg for orders assigned to a drone 'k' must not exceed DRONE_MAX_PAYLOAD_KG.
#    - Battery Constraint: The total energy cost for the route of drone 'k' must not exceed DRONE_BATTERY_WH.
#
# This makes the problem significantly more complex but is a natural extension of this QUBO-based approach for fleet-level optimization.