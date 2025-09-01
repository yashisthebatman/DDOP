import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
import config
from .geometry import calculate_wind_effect

GridCoord = Tuple[int, int, int]
PathPlanner = 'PathPlanner3D'

class Heuristic(ABC):
    def __init__(self, planner: PathPlanner):
        self.planner = planner
        self.payload_kg = 0.0
        self.goal = (0, 0, 0)
        self.weight = config.A_STAR_HEURISTIC_WEIGHT
        self.time_weight = 0.5
        self.energy_weight = 0.5

    def update_params(self, payload_kg: float, goal: GridCoord, time_weight: float = 0.5):
        self.payload_kg = payload_kg
        self.goal = goal
        self.time_weight = time_weight
        self.energy_weight = 1.0 - time_weight

    @abstractmethod
    def calculate(self, node: GridCoord) -> float:
        """Calculate heuristic value from node to goal."""
        pass

    def cost_between(self, n1: GridCoord, n2: GridCoord) -> float:
        """Calculate actual cost between adjacent nodes."""
        if self.planner.is_grid_obstructed(n2):
            return float('inf')
            
        cost_multiplier = self.planner.cost_map.get(n2, config.DEFAULT_CELL_COST)
        dist = np.linalg.norm(np.array(n1) - np.array(n2))
        return dist * cost_multiplier

class TimeHeuristic(Heuristic):
    def __init__(self, planner: PathPlanner):
        super().__init__(planner)

    def calculate(self, node: GridCoord) -> float:
        dist_grid = np.linalg.norm(np.array(node) - np.array(self.goal))
        return dist_grid * self.weight

class EnergyHeuristic(Heuristic):
    def __init__(self, planner: PathPlanner):
        super().__init__(planner)

    def calculate(self, node: GridCoord) -> float:
        node_world = self.planner._grid_to_world(node)
        goal_world = self.planner._grid_to_world(self.goal)
        dist_world = np.linalg.norm(np.array(goal_world) - np.array(node_world))
        
        base_energy = dist_world * getattr(self.planner, 'baseline_energy_per_meter', 0.01)
        alt_change = goal_world[2] - node_world[2]
        
        p_energy = 0.0
        if alt_change > 0:
            total_mass = config.DRONE_MASS_KG + self.payload_kg
            joules = (total_mass * config.GRAVITY * alt_change) / config.ASCENT_EFFICIENCY
            p_energy = joules / 3600

        return self.weight * (base_energy + p_energy)

class BalancedHeuristic(Heuristic):
    def __init__(self, planner: PathPlanner):
        super().__init__(planner)
        self.time_h = TimeHeuristic(planner)
        self.energy_h = EnergyHeuristic(planner)
        self.max_time_est = 3600
        self.max_energy_est = config.DRONE_BATTERY_WH

    def update_params(self, payload_kg: float, goal: GridCoord, time_weight: float = 0.5):
        super().update_params(payload_kg, goal, time_weight)
        self.time_h.update_params(payload_kg, goal, time_weight)
        self.energy_h.update_params(payload_kg, goal, time_weight)

    def calculate(self, node: GridCoord) -> float:
        time_cost = self.time_h.calculate(node) / self.max_time_est
        energy_cost = self.energy_h.calculate(node) / self.max_energy_est
        
        combined = (self.time_weight * time_cost + self.energy_weight * energy_cost)
        return combined * self.weight