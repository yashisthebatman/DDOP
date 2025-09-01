# utils/heuristics.py (Full and Corrected)

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

import config
from .geometry import calculate_wind_effect

# Type Aliases
GridCoord = Tuple[int, int, int]
PathPlanner = 'PathPlanner3D'

class Heuristic(ABC):
    def __init__(self, planner: PathPlanner):
        """
        CORRECTED: Initialized with only the planner. 
        Mission-specific parameters are set later.
        """
        self.planner = planner
        self.payload_kg = 0.0
        self.goal = (0, 0, 0)
        self.weight = config.A_STAR_HEURISTIC_WEIGHT
        self.time_weight = 0.5
        self.energy_weight = 0.5

    def update_params(self, payload_kg: float, goal: GridCoord, time_weight: float = 0.5):
        """NEW: Sets the mission-specific parameters before a search."""
        self.payload_kg = payload_kg
        self.goal = goal
        self.time_weight = time_weight
        self.energy_weight = 1.0 - time_weight

    @abstractmethod
    def calculate(self, node: GridCoord) -> float:
        """Calculates the heuristic value from a node to the goal."""
        pass

    def cost_between(self, n1: GridCoord, n2: GridCoord) -> float:
        """Calculates the actual cost of moving between two adjacent nodes."""
        # This part remains unchanged
        base_cost = np.linalg.norm(np.array(n1) - np.array(n2))
        cost_multiplier = self.planner.cost_map.get(n2, 1.0) # Default cost is 1.0
        return base_cost * cost_multiplier

class TimeHeuristic(Heuristic):
    def calculate(self, node: GridCoord) -> float:
        dist = np.linalg.norm(np.array(self.goal) - np.array(node)) * self.planner.grid_size
        est_time = dist / config.DRONE_SPEED_MPS
        return est_time * self.weight

class EnergyHeuristic(Heuristic):
    def calculate(self, node: GridCoord) -> float:
        node_world = self.planner._grid_to_world(node)
        goal_world = self.planner._grid_to_world(self.goal)
        dist = np.linalg.norm(np.array(goal_world) - np.array(node_world))
        
        # Horizontal energy
        h_energy = dist * (50 + self.payload_kg * 10) / 3600

        # Potential energy
        alt_change = goal_world[2] - node_world[2]
        p_energy = 0.0
        if alt_change > 0:
            total_mass = config.DRONE_MASS_KG + self.payload_kg
            joules = (total_mass * config.GRAVITY * alt_change) / config.ASCENT_EFFICIENCY
            p_energy = joules / 3600
            
        return (h_energy + p_energy) * self.weight

class BalancedHeuristic(Heuristic):
    def __init__(self, planner: PathPlanner):
        super().__init__(planner)
        self.time_h = TimeHeuristic(planner)
        self.energy_h = EnergyHeuristic(planner)
        # Normalization factors, can be rough estimates
        self.max_time_est = 3600 # An upper bound for time in seconds
        self.max_energy_est = config.DRONE_BATTERY_WH # An upper bound for energy

    def update_params(self, payload_kg: float, goal: GridCoord, time_weight: float = 0.5):
        super().update_params(payload_kg, goal, time_weight)
        # Update the child heuristics as well
        self.time_h.update_params(payload_kg, goal, time_weight)
        self.energy_h.update_params(payload_kg, goal, time_weight)

    def calculate(self, node: GridCoord) -> float:
        time_c = self.time_h.calculate(node) / self.max_time_est
        energy_c = self.energy_h.calculate(node) / self.max_energy_est
        return (self.time_weight * time_c + self.energy_weight * energy_c) * self.weight