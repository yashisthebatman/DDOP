# optimization/assignment_solver.py
import numpy as np
import logging
import config

class AssignmentSolver:
    def __init__(self, env, predictor, path_planner, path_solver_choice):
        self.env = env
        self.predictor = predictor
        self.path_planner = path_planner
        self.path_solver_choice = path_solver_choice
        self.trip_costs = self._precompute_trip_costs()

    def _precompute_trip_costs(self):
        costs = {}
        hub = config.HUB_LOCATION
        for order in self.env.orders:
            logging.info(f"Planning path for order {order.id} with {self.path_solver_choice}...")
            path_to = self.path_planner.find_path(hub, order.location, order.payload_kg, self.path_solver_choice)
            path_from = self.path_planner.find_path(order.location, hub, 0, self.path_solver_choice)
            
            if path_to is None or path_from is None:
                logging.warning(f"Could not find a complete path for order {order.id} using {self.path_solver_choice}.")
                costs[order.id] = {'time': float('inf'), 'energy': float('inf'), 'path_to': [], 'path_from': []}
                continue
            
            time_to, energy_to = self._calculate_path_cost(path_to, order.payload_kg)
            time_from, energy_from = self._calculate_path_cost(path_from, 0)

            total_trip_time = time_to + time_from + config.RECHARGE_TIME_S
            total_trip_energy = energy_to + energy_from

            logging.info(f"Order {order.id}: Time={total_trip_time:.2f}s, Energy={total_trip_energy:.2f}Wh")

            costs[order.id] = {
                'time': total_trip_time,
                'energy': total_trip_energy,
                'path_to': path_to,
                'path_from': path_from
            }
        return costs

    def _calculate_path_cost(self, path, payload_kg):
        total_time, total_energy = 0, 0
        if not path: return 0,0
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            wind = self.env.weather.get_wind_at_location(p1[0], p1[1])
            t, e = self.predictor.predict(p1, p2, payload_kg, wind)
            total_time += t
            total_energy += e
        return total_time, total_energy

    def solve(self, weights):
        assignments = {d.id: [] for d in self.env.drones}
        drone_finish_times = {d.id: 0.0 for d in self.env.drones}
        
        # Filter out orders for which no path could be found
        possible_orders = [o for o in self.env.orders if self.trip_costs[o.id]['time'] != float('inf')]
        
        if not possible_orders:
            logging.warning("No orders could be assigned as no valid paths were found for any of them.")
            return self._format_solution(assignments)

        def order_cost(order):
            costs = self.trip_costs[order.id]
            return weights['time'] * costs['time'] + weights['energy'] * costs['energy']
            
        # Sort the plannable orders by their weighted cost
        sorted_orders = sorted(possible_orders, key=order_cost)

        # Simple greedy assignment: give the next cheapest order to the next available drone
        for order in sorted_orders:
            best_drone_id = min(drone_finish_times, key=drone_finish_times.get)
            start_time = drone_finish_times[best_drone_id]
            duration = self.trip_costs[order.id]['time']
            end_time = start_time + duration
            
            task = {'order_id': order.id, 'start_time': start_time, 'end_time': end_time, 'duration': duration}
            assignments[best_drone_id].append(task)
            drone_finish_times[best_drone_id] = end_time

        return self._format_solution(assignments)

    def _format_solution(self, assignments):
        total_energy, max_time = 0, 0
        full_paths = {d.id: [] for d in self.env.drones}
        
        for d_id, tasks in assignments.items():
            tasks.sort(key=lambda t: t['start_time'])
            for task in tasks:
                o_id = task['order_id']
                costs = self.trip_costs[o_id]
                total_energy += costs['energy']
                max_time = max(max_time, task['end_time'])
                
                if costs['path_to'] and costs['path_from']:
                    combined_path = costs['path_to'] + costs['path_from'][1:]
                else:
                    combined_path = costs['path_to'] or costs['path_from']

                full_paths[d_id].append({
                    'start_time': task['start_time'],
                    'end_time': task['end_time'],
                    'duration': task['duration'],
                    'path': combined_path
                })
        return {'assignments': assignments, 'full_paths': full_paths, 'total_time': max_time, 'total_energy': total_energy, 'env': self.env}