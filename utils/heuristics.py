# utils/heuristics.py
import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod

import config
from .geometry import calculate_wind_effect

# Type Aliases
GridCoord = Tuple[int, int, int]
PathPlanner = 'PathPlanner3D' # Forward declaration

class Heuristic(ABC):
    """Abstract base class for all heuristic implementations."""
    def __init__(self, planner: PathPlanner, payload_kg: float, goal: GridCoord, time_weight: float = 0.5):
        self.planner = planner
        self.payload_kg = payload_kg
        self.goal = goal
        self.weight = config.A_STAR_HEURISTIC_WEIGHT
        self.time_weight = time_weight
        self.energy_weight = 1.0 - time_weight

    @abstractmethod
    def calculate(self, node: GridCoord) -> float:
        """Calculates the heuristic value (estimated cost) from a given node to the goal."""
        pass

    @abstractmethod
    def cost_between(self, n1: GridCoord, n2: GridCoord, p_prev: Optional[GridCoord]) -> float:
        """Calculates the actual cost of moving between two adjacent nodes."""
        pass

class TimeHeuristic(Heuristic):
    """Estimates and calculates the time cost, considering wind."""
    def calculate(self, node: GridCoord) -> float:
        node_world = self.planner._grid_to_world(node)
        goal_world = self.planner._grid_to_world(self.goal)
        distance_m = np.linalg.norm(np.array(goal_world) - np.array(node_world))
        if distance_m < 1e-6: return 0.0

        avg_wind = self.planner.env.weather.get_wind_at_location(node_world[0], node_world[1])
        flight_vector = np.array(goal_world) - np.array(node_world)
        time_impact, _ = calculate_wind_effect(flight_vector, avg_wind, config.DRONE_SPEED_MPS)
        
        estimated_cost = (distance_m * self.planner.baseline_time_per_meter) * time_impact
        return self.weight * estimated_cost

    def cost_between(self, n1: GridCoord, n2: GridCoord, p_prev: Optional[GridCoord]) -> float:
        w1 = self.planner._grid_to_world(n1)
        w2 = self.planner._grid_to_world(n2)
        w_prev = self.planner._grid_to_world(p_prev) if p_prev else None
        wind = self.planner.env.weather.get_wind_at_location(w1[0], w1[1])
        time, _ = self.planner.predictor.predict(w1, w2, self.payload_kg, wind, w_prev)
        return time if time != float('inf') else float('inf')

class EnergyHeuristic(Heuristic):
    """Estimates and calculates the energy cost, considering altitude, payload, and wind."""
    def calculate(self, node: GridCoord) -> float:
        node_world = self.planner._grid_to_world(node)
        goal_world = self.planner._grid_to_world(self.goal)
        distance_m = np.linalg.norm(np.array(goal_world) - np.array(node_world))
        
        # Estimate wind's energy impact
        avg_wind = self.planner.env.weather.get_wind_at_location(node_world[0], node_world[1])
        flight_vector = np.array(goal_world) - np.array(node_world)
        _, energy_impact = calculate_wind_effect(flight_vector, avg_wind, config.DRONE_SPEED_MPS)

        horizontal_energy_cost = distance_m * self.planner.baseline_energy_per_meter * energy_impact

        # Estimate potential energy cost for altitude change
        altitude_change_m = goal_world[2] - node_world[2]
        potential_energy_cost = 0.0
        if altitude_change_m > 0:
            total_mass = config.DRONE_MASS_KG + self.payload_kg
            joules = (total_mass * config.GRAVITY * altitude_change_m) / config.ASCENT_EFFICIENCY
            potential_energy_wh = joules / 3600
            potential_energy_cost = potential_energy_wh # Already in the right "units" of cost
            
        estimated_cost = horizontal_energy_cost + potential_energy_cost
        return self.weight * estimated_cost

    def cost_between(self, n1: GridCoord, n2: GridCoord, p_prev: Optional[GridCoord]) -> float:
        w1 = self.planner._grid_to_world(n1)
        w2 = self.planner._grid_to_world(n2)
        w_prev = self.planner._grid_to_world(p_prev) if p_prev else None
        wind = self.planner.env.weather.get_wind_at_location(w1[0], w1[1])
        _, energy = self.planner.predictor.predict(w1, w2, self.payload_kg, wind, w_prev)
        return energy if energy != float('inf') else float('inf')

class BalancedHeuristic(Heuristic):
    """A normalized blend of the time and energy heuristics for true balanced optimization."""
    def __init__(self, planner: PathPlanner, payload_kg: float, goal: GridCoord, time_weight: float = 0.5):
        super().__init__(planner, payload_kg, goal, time_weight)
        self.time_heuristic = TimeHeuristic(planner, payload_kg, goal, time_weight)
        self.energy_heuristic = EnergyHeuristic(planner, payload_kg, goal, time_weight)

        # Establish normalization factors based on a straight-line flight estimate
        start_world = planner._grid_to_world(goal) # Placeholder for any node
        goal_world = planner._grid_to_world(goal)
        straight_dist = np.linalg.norm(np.array(start_world) - np.array(goal_world))
        self.max_time_estimate = (straight_dist / config.DRONE_SPEED_MPS) * 2 + 1 # Add buffer
        self.max_energy_estimate = (straight_dist * planner.baseline_energy_per_meter) * 2 + 1 # Add buffer

    def calculate(self, node: GridCoord) -> float:
        time_h = self.time_heuristic.calculate(node) / self.max_time_estimate
        energy_h = self.energy_heuristic.calculate(node) / self.max_energy_estimate
        # The sub-heuristics already applied the A* weight, so we just balance them
        return (self.time_weight * time_h) + (self.energy_weight * energy_h)

    def cost_between(self, n1: GridCoord, n2: GridCoord, p_prev: Optional[GridCoord]) -> float:
        time_cost = self.time_heuristic.cost_between(n1, n2, p_prev)
        energy_cost = self.energy_heuristic.cost_between(n1, n2, p_prev)
        if time_cost == float('inf') or energy_cost == float('inf'):
            return float('inf')

        norm_time = time_cost / self.max_time_estimate
        norm_energy = energy_cost / self.max_energy_estimate
        return (self.time_weight * norm_time) + (self.energy_weight * norm_energy)