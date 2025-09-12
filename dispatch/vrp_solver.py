# FILE: dispatch/vrp_solver.py
import logging
from typing import List, Dict
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

from ml_predictor.predictor import EnergyTimePredictor
from config import DRONE_MAX_PAYLOAD_KG

class VRPSolver:
    """Uses Google OR-Tools to solve the Vehicle Routing Problem for drone delivery."""

    def __init__(self, predictor: EnergyTimePredictor):
        self.predictor = predictor

    def _create_data_model(self, drones: List[Dict], orders: List[Dict]) -> Dict:
        """Prepares the data for the VRP solver."""
        locations = [drone['pos'] for drone in drones] + [order['pos'] for order in orders]
        
        # Create a cost matrix (time + energy weighted)
        num_locations = len(locations)
        cost_matrix = np.zeros((num_locations, num_locations))

        for from_node in range(num_locations):
            for to_node in range(num_locations):
                if from_node == to_node:
                    continue
                p1, p2 = locations[from_node], locations[to_node]
                # Assume average payload and no wind for routing cost estimation
                payload = DRONE_MAX_PAYLOAD_KG / 2 
                wind = [0, 0, 0]
                time, energy = self.predictor.predict(p1, p2, payload, wind, None)
                # Weighted cost: simple sum for now, can be tuned. OR-Tools prefers integer costs.
                cost_matrix[from_node, to_node] = int((time + energy) * 10)
        
        num_vehicles = len(drones)
        depot_indices = list(range(num_vehicles))

        data = {
            'cost_matrix': cost_matrix.tolist(),
            'demands': [0] * num_vehicles + [int(o['payload_kg'] * 100) for o in orders], # Use integer demands
            'vehicle_capacities': [int(d['max_payload_kg'] * 100) for d in drones],
            'num_vehicles': num_vehicles,
            'starts': depot_indices,
            'ends': depot_indices
        }
        return data

    def generate_tours(self, drones: List[Dict], orders: List[Dict]) -> List[Dict]:
        """
        Solves the VRP to generate optimal delivery tours for the given drones and orders.
        Returns a list of tours, where each tour is a dictionary.
        """
        if not drones or not orders:
            return []

        # Map drone and order objects for easy lookup later
        drone_map = {i: d for i, d in enumerate(drones)}
        order_map = {i + len(drones): o for i, o in enumerate(orders)}

        data = self._create_data_model(drones, orders)
        
        manager = pywrapcp.RoutingIndexManager(
            len(data['cost_matrix']), data['num_vehicles'], data['starts'], data['ends']
        )
        routing = pywrapcp.RoutingModel(manager)

        # Cost callback
        def cost_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['cost_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(cost_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Payload capacity constraint
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],
            True,  # start cumul to zero
            'Capacity'
        )

        # Setting first solution heuristic
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(5)

        logging.info("Solving VRP to generate delivery tours...")
        solution = routing.SolveWithParameters(search_parameters)
        
        if not solution:
            logging.warning("VRP solver found no solution.")
            return []

        tours = []
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            tour_stops = []
            tour_payload = 0
            
            index = solution.Value(routing.NextVar(index))
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                order = order_map[node_index]
                tour_stops.append(order)
                tour_payload += order['payload_kg']
                index = solution.Value(routing.NextVar(index))
            
            if tour_stops:
                tours.append({
                    'drone_id': drone_map[vehicle_id]['id'],
                    'stops': tour_stops,
                    'payload': round(tour_payload, 2)
                })
        
        logging.info(f"VRP solver generated {len(tours)} valid tours.")
        return tours