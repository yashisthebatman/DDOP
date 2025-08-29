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
            path_to = self.path_planner.find_path(hub, order.location, order.payload_kg, self.path_solver_choice)
            path_from = self.path_planner.find_path(order.location, hub, 0, self.path_solver_choice)
            
            if path_to is None or path_from is None:
                logging.warning(f"Could not find path for order {order.id} using {self.path_solver_choice}.")
                costs[order.id] = {'time': float('inf'), 'energy': float('inf'), 'path_to': [], 'path_from': []}
                continue
            
            time_to, energy_to = self._calculate_path_cost(path_to, order.payload_kg)
            time_from, energy_from = self._calculate_path_cost(path_from, 0)

            costs[order.id] = {
                'time': time_to + time_from + config.RECHARGE_TIME_S,
                'energy': energy_to + energy_from,
                'path_to': path_to,
                'path_from': path_from
            }
        return costs

    def _calculate_path_cost(self, path, payload_kg):
        total_time, total_energy = 0, 0
        for i in range(len(path) - 1):
            wind = self.env.weather.get_wind_at_location(path[i][0], path[i][1])
            t, e = self.predictor.predict(path[i], path[i+1], payload_kg, wind)
            total_time += t
            total_energy += e
        return total_time, total_energy

    def solve(self, weights):
        assignments = {d.id: [] for d in self.env.drones}
        drone_finish_times = {d.id: 0.0 for d in self.env.drones}
        
        possible_orders = [o for o in self.env.orders if self.trip_costs[o.id]['time'] != float('inf')]
        
        def order_cost(order):
            costs = self.trip_costs[order.id]
            return weights['time'] * costs['time'] + weights['energy'] * costs['energy']
            
        sorted_orders = sorted(possible_orders, key=order_cost)

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
                
                # Robustly combine paths
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