import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

import config
from .geometry import calculate_wind_effect

# Type Aliases
GridCoord = Tuple[int, int, int]
PathPlanner = 'PathPlanner3D' # Forward declaration for type hinting

class Heuristic(ABC):
    """Abstract base class for all heuristic implementations."""
    def __init__(self, planner: PathPlanner, payload_kg: float, goal: GridCoord):
        self.planner = planner
        self.payload_kg = payload_kg
        self.goal = goal
        self.weight = config.A_STAR_HEURISTIC_WEIGHT

    @abstractmethod
    def calculate(self, node: GridCoord) -> float:
        """Calculates the heuristic value from a given node to the goal."""
        pass

class TimeHeuristic(Heuristic):
    """Estimates the time cost to reach the goal, considering wind."""
    def calculate(self, node: GridCoord) -> float:
        node_world = self.planner._grid_to_world(node)
        goal_world = self.planner._grid_to_world(self.goal)
        
        distance_m = np.linalg.norm(np.array(goal_world) - np.array(node_world))
        if distance_m < 1e-6: return 0.0

        # Estimate wind effect for the remainder of the journey
        avg_wind = self.planner.env.weather.get_wind_at_location(node_world[0], node_world[1])
        flight_vector = np.array(goal_world) - np.array(node_world)
        time_impact, _ = calculate_wind_effect(flight_vector, avg_wind, config.DRONE_SPEED_MPS)
        
        estimated_cost = (distance_m * self.planner.baseline_time_per_meter) * time_impact
        return self.weight * estimated_cost

class EnergyHeuristic(Heuristic):
    """Estimates the energy cost to reach the goal, considering altitude and payload."""
    def calculate(self, node: GridCoord) -> float:
        distance_m = np.linalg.norm(np.array(self.goal) - np.array(node)) * self.planner.resolution
        
        # Estimate potential energy cost for altitude change
        altitude_change_m = (self.goal[2] - node[2]) * self.planner.resolution
        potential_energy_cost = 0.0
        if altitude_change_m > 0:
            total_mass = config.DRONE_MASS_KG + self.payload_kg
            joules = (total_mass * config.GRAVITY * altitude_change_m) / config.ASCENT_EFFICIENCY
            # Scale potential energy to match the units/scale of the horizontal energy estimate
            potential_energy_wh = joules / 3600
            potential_energy_cost = potential_energy_wh * self.planner.baseline_energy_per_meter * 10 # Empirical scaler

        horizontal_energy_cost = distance_m * self.planner.baseline_energy_per_meter
        estimated_cost = horizontal_energy_cost + potential_energy_cost
        return self.weight * estimated_cost

class BalancedHeuristic(Heuristic):
    """A blend of the time and energy heuristics for balanced optimization."""
    def __init__(self, planner: PathPlanner, payload_kg: float, goal: GridCoord):
        super().__init__(planner, payload_kg, goal)
        # Internally create instances of the other heuristics to combine them
        self.time_heuristic = TimeHeuristic(planner, payload_kg, goal)
        self.energy_heuristic = EnergyHeuristic(planner, payload_kg, goal)

    def calculate(self, node: GridCoord) -> float:
        time_cost = self.time_heuristic.calculate(node)
        energy_cost = self.energy_heuristic.calculate(node)
        # We don't apply the weight here, as it's already applied in the sub-heuristics
        return time_cost + energy_cost