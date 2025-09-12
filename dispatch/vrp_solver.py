# FILE: dispatch/vrp_solver.py
import logging
from typing import List, Dict
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

from ml_predictor.predictor import EnergyTimePredictor
from config import DRONE_MAX_PAYLOAD_KG, HUBS

class VRPSolver:
    """Uses Google OR-Tools to solve the Multi-Depot Vehicle Routing Problem."""

    def __init__(self, predictor: EnergyTimePredictor):
        self.predictor = predictor

    def _create_data_model(self, drones: List[Dict], orders: List[Dict]) -> Dict:
        """Prepares the data for the Multi-Depot VRP solver."""
        hub_names = list(HUBS.keys())
        hub_map = {name: i for i, name in enumerate(hub_names)}
        
        all_hubs_pos = list(HUBS.values())
        all_orders_pos = [order['pos'] for order in orders]
        locations = all_hubs_pos + all_orders_pos

        num_hubs = len(all_hubs_pos)
        num_locations = len(locations)
        sink_node = num_locations # A virtual node for flexible mission ends

        cost_matrix = np.full((num_locations + 1, num_locations + 1), 1_000_000)

        for from_node in range(num_locations):
            for to_node in range(num_locations):
                if from_node == to_node:
                    cost_matrix[from_node, to_node] = 0
                    continue
                p1, p2 = locations[from_node], locations[to_node]
                payload = DRONE_MAX_PAYLOAD_KG / 2
                wind = [0, 0, 0]
                time, energy = self.predictor.predict(p1, p2, payload, wind, None)
                cost_matrix[from_node, to_node] = int((time + energy) * 10)
        
        # Allow travel from any location that is a hub to the sink node at zero cost
        for to_node_idx in range(num_hubs):
             cost_matrix[to_node_idx, sink_node] = 0

        num_vehicles = len(drones)
        starts = [hub_map[d['home_hub']] for d in drones]
        ends = [sink_node] * num_vehicles

        data = {
            'cost_matrix': cost_matrix.tolist(),
            'demands': [0] * num_hubs + [int(o['payload_kg'] * 100) for o in orders] + [0], # Hubs/sink have 0 demand
            'vehicle_capacities': [int(d['max_payload_kg'] * 100) for d in drones],
            'num_vehicles': num_vehicles,
            'starts': starts,
            'ends': ends,
            'num_locations': num_locations + 1,
            'hub_names': hub_names
        }
        return data

    def generate_tours(self, drones: List[Dict], orders: List[Dict]) -> List[Dict]:
        """
        Solves the MDVRP to generate optimal delivery tours for the given drones and orders.
        Returns a list of tours, where each tour is a dictionary.
        """
        if not drones or not orders:
            return []

        drone_map = {i: d for i, d in enumerate(drones)}
        order_map = {i + len(HUBS): o for i, o in enumerate(orders)}

        data = self._create_data_model(drones, orders)
        
        manager = pywrapcp.RoutingIndexManager(
            data['num_locations'], data['num_vehicles'], data['starts'], data['ends']
        )
        routing = pywrapcp.RoutingModel(manager)

        def cost_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['cost_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(cost_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity'
        )

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(5)

        logging.info("Solving MDVRP to generate delivery tours...")
        solution = routing.SolveWithParameters(search_parameters)
        
        if not solution:
            logging.warning("VRP solver found no solution.")
            return []

        tours = []
        num_hubs = len(data['hub_names'])
        
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            tour_stops = []
            tour_payload = 0
            
            route = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                index = solution.Value(routing.NextVar(index))
            
            if len(route) <= 2: continue # Only start and (virtual) end node

            # The end hub is the node visited just before the sink node
            last_real_node = route[-2]
            if last_real_node >= num_hubs: # If the last stop is an order, find the next hub
                end_hub_pos = HUBS[min(HUBS, key=lambda h: np.linalg.norm(np.array(HUBS[h]) - np.array(order_map[last_real_node]['pos'])))]
                end_hub_id = [k for k,v in HUBS.items() if v == end_hub_pos][0]

            elif last_real_node < num_hubs:
                end_hub_id = data['hub_names'][last_real_node]
            else:
                logging.warning(f"Could not determine end hub for drone {drone_map[vehicle_id]['id']}. Skipping.")
                continue
            
            order_nodes = [node for node in route if node >= num_hubs and node < data['num_locations'] - 1]
            for node_index in order_nodes:
                order = order_map[node_index]
                tour_stops.append(order)
                tour_payload += order['payload_kg']
            
            if tour_stops:
                tours.append({
                    'drone_id': drone_map[vehicle_id]['id'],
                    'start_hub_id': drone_map[vehicle_id]['home_hub'],
                    'end_hub_id': end_hub_id,
                    'stops': tour_stops,
                    'payload': round(tour_payload, 2)
                })
        
        logging.info(f"VRP solver generated {len(tours)} valid tours.")
        return tours