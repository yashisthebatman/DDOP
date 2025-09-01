import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

import config
from .geometry import calculate_wind_effect

# Type Aliases
GridCoord = Tuple[int, int, int]
PathPlanner = 'PathPlanner3D'

class Heuristic(ABC):
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

    def cost_between(self, n1: GridCoord, n2: GridCoord) -> float:
        """
        Calculates the actual cost of moving between two adjacent nodes.
        This now checks for obstacles, making it suitable for D* Lite.
        """
        # 1. Check if the destination node is obstructed
        if self.planner.is_grid_obstructed(n2):
            return float('inf')
            
        # 2. Check the pre-computed environmental cost map (for wind, etc.)
        cost_multiplier = self.planner.cost_map.get(n2, 1.0) # Default cost is 1.0

        # 3. Calculate geometric distance (in grid units)
        dist = np.linalg.norm(np.array(n1) - np.array(n2))
        
        # Total cost is distance scaled by environmental factors
        return dist * cost_multiplier

class TimeHeuristic(Heuristic):
    def calculate(self, node: GridCoord) -> float:
        dist_grid = np.linalg.norm(np.array(node) - np.array(self.goal))
        # Use simple grid distance for performance in low-level search
        return dist_grid * self.weight

class EnergyHeuristic(Heuristic):
    def calculate(self, node: GridCoord) -> float:
        # Complex heuristic for energy is slow; simpler version for low-level search
        node_world = self.planner._grid_to_world(node)
        goal_world = self.planner._grid_to_world(self.goal)
        dist_world = np.linalg.norm(np.array(goal_world) - np.array(node_world))
        
        base_energy = dist_world * self.planner.baseline_energy_per_meter
        alt_change = goal_world[2] - node_world[2]
        p_energy = 0.0
        if alt_change > 0:
            total_mass = config.DRONE_MASS_KG + self.payload_kg
            joules = (total_mass * config.GRAVITY * alt_change) / config.ASCENT_EFFICIENCY
            p_energy = joules / 3600

        return self.weight * (base_energy + p_energy)


class BalancedHeuristic(Heuristic):
    def calculate(self, node: GridCoord) -> float:
        # A simple Euclidean distance heuristic is fast and effective for both JPS and D* Lite
        dist = np.linalg.norm(np.array(node) - np.array(self.goal))
        return dist * self.weight