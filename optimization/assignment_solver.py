# optimization/assignment_solver.py
import numpy as np
from ortools.sat.python import cp_model

import config

class AssignmentSolver:
    def __init__(self, env, predictor, path_planner):
        self.env = env
        self.predictor = predictor
        self.path_planner = path_planner
        self.trip_costs = self._precompute_trip_costs()

    def _precompute_trip_costs(self):
        """Calculates the time and energy for every possible round trip."""
        costs = {}
        hub = config.HUB_LOCATION
        for order in self.env.orders:
            path_to = self.path_planner.find_path(hub, order.location)
            path_from = self.path_planner.find_path(order.location, hub)
            
            if path_to is None or path_from is None:
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
            t, e = self.predictor.predict(path[i], path[i+1], payload_kg, self.env.wind_vector)
            total_time += t
            total_energy += e
        return total_time, total_energy

    def solve_with_or_tools(self, weights):
        """Solves the assignment problem using Google OR-Tools CP-SAT solver."""
        model = cp_model.CpModel()
        
        x = {(d.id, o.id): model.NewBoolVar(f'x_{d.id}_{o.id}')
             for d in self.env.drones for o in self.env.orders}

        for o in self.env.orders:
            model.AddExactlyOne(x[d.id, o.id] for d in self.env.drones)

        task_intervals = {}
        for d in self.env.drones:
            drone_tasks = []
            for o in self.env.orders:
                # --- THIS IS THE FIX: Handle impossible tasks ---
                duration = self.trip_costs[o.id]['time']
                if duration == float('inf'):
                    # This order is unreachable. Forbid its assignment to this drone.
                    model.Add(x[d.id, o.id] == 0)
                    continue # Skip creating an interval for this impossible task
                
                int_duration = int(duration)
                # --- END OF FIX ---
                start = model.NewIntVar(0, 100000, f'start_{d.id}_{o.id}')
                end = model.NewIntVar(0, 100000, f'end_{d.id}_{o.id}')
                interval = model.NewOptionalIntervalVar(start, int_duration, end, x[d.id, o.id], f'interval_{d.id}_{o.id}')
                drone_tasks.append(interval)
                task_intervals[d.id, o.id] = interval
            model.AddNoOverlap(drone_tasks)

        makespan = model.NewIntVar(0, 100000, 'makespan')
        for d in self.env.drones:
            for o in self.env.orders:
                if self.trip_costs[o.id]['time'] != float('inf'):
                    model.Add(makespan >= task_intervals[d.id, o.id].EndExpr()).OnlyEnforceIf(x[d.id, o.id])
        
        # Also fix the energy summation to ignore impossible tasks
        possible_tasks_energy = [int(self.trip_costs[o.id]['energy']) * x[d.id, o.id] 
                                 for d in self.env.drones for o in self.env.orders 
                                 if self.trip_costs[o.id]['energy'] != float('inf')]
        total_energy_cost = model.NewIntVar(0, 1000000, 'total_energy')
        model.Add(total_energy_cost == sum(possible_tasks_energy))
        
        objective = (weights['time'] * 100 * makespan) + (weights['energy'] * total_energy_cost)
        model.Minimize(objective)
        
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._format_solution(solver, x, task_intervals)
        else:
            return {'assignments': {}, 'full_paths': {}, 'total_time': 0, 'total_energy': 0, 'env': self.env}

    def solve_with_qubo(self, weights):
        """Placeholder using a greedy algorithm that mimics a QUBO assignment."""
        assignments = {d.id: [] for d in self.env.drones}
        drone_finish_times = {d.id: 0 for d in self.env.drones}
        
        # Proactive fix: Filter out impossible orders before assigning
        possible_orders = [o for o in self.env.orders if self.trip_costs[o.id]['time'] != float('inf')]
        
        for order in possible_orders:
            best_drone_id = min(drone_finish_times, key=drone_finish_times.get)
            start_time = drone_finish_times[best_drone_id]
            duration = self.trip_costs[order.id]['time']
            end_time = start_time + duration
            assignments[best_drone_id].append({'order_id': order.id, 'start_time': start_time, 'end_time': end_time, 'duration': duration})
            drone_finish_times[best_drone_id] = end_time
        return self._format_greedy_solution(assignments)

    def _format_solution(self, solver, x, task_intervals):
        assignments = {d.id: [] for d in self.env.drones}
        for d in self.env.drones:
            for o in self.env.orders:
                if self.trip_costs[o.id]['time'] != float('inf') and solver.Value(x[d.id, o.id]):
                    interval = task_intervals[d.id, o.id]
                    assignments[d.id].append({
                        'order_id': o.id,
                        'start_time': solver.Value(interval.StartExpr()),
                        'end_time': solver.Value(interval.EndExpr()),
                        'duration': solver.Value(interval.EndExpr()) - solver.Value(interval.StartExpr())
                    })
        return self._format_greedy_solution(assignments)

    def _format_greedy_solution(self, assignments):
        total_energy, max_time = 0, 0
        full_paths = {d.id: [] for d in self.env.drones}
        
        for d_id, tasks in assignments.items():
            tasks.sort(key=lambda t: t['start_time'])
            for task in tasks:
                o_id = task['order_id']
                costs = self.trip_costs[o_id]
                total_energy += costs['energy']
                max_time = max(max_time, task['end_time'])
                full_paths[d_id].append({
                    'start_time': task['start_time'],
                    'end_time': task['end_time'],
                    'duration': task['duration'],
                    'path': costs['path_to'] + costs['path_from'][1:]
                })
        return {'assignments': assignments, 'full_paths': full_paths, 'total_time': max_time, 'total_energy': total_energy, 'env': self.env}