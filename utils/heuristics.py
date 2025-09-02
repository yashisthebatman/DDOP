# ==============================================================================
# File: utils/heuristics.py
# ==============================================================================
import numpy as np
from typing import Tuple, TYPE_CHECKING
from abc import ABC, abstractmethod

from config import (
    A_STAR_HEURISTIC_WEIGHT, DEFAULT_CELL_COST, DRONE_MASS_KG, GRAVITY,
    ASCENT_EFFICIENCY, DRONE_BATTERY_WH, DRONE_SPEED_MPS
)

# Use TYPE_CHECKING to avoid circular import errors
if TYPE_CHECKING:
    from path_planner import PathPlanner3D

GridCoord = Tuple[int, int, int]

class Heuristic(ABC):
    def __init__(self, planner: 'PathPlanner3D'):
        self.planner = planner
        self.payload_kg = 0.0
        self.goal = (0, 0, 0)
        self.weight = A_STAR_HEURISTIC_WEIGHT
        self.time_weight = 0.5
        self.energy_weight = 0.5

    def update_params(self, payload_kg: float, goal: GridCoord, time_weight: float = 0.5, **kwargs):
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
            
        cost_multiplier = self.planner.cost_map.get(n2, DEFAULT_CELL_COST)
        dist = np.linalg.norm(np.array(n1) - np.array(n2))
        return dist * cost_multiplier

class TimeHeuristic(Heuristic):
    def calculate(self, node: GridCoord) -> float:
        # Heuristic in grid space is a direct distance calculation
        dist_grid = np.linalg.norm(np.array(node) - np.array(self.goal))
        return dist_grid * self.weight

class EnergyHeuristic(Heuristic):
    def calculate(self, node: GridCoord) -> float:
        # Use the planner's coordinate manager for all conversions
        node_world = self.planner.coord_manager.grid_to_world(node)
        goal_world = self.planner.coord_manager.grid_to_world(self.goal)
        dist_world = np.linalg.norm(np.array(goal_world) - np.array(node_world))
        
        # Estimate energy based on distance and potential energy change
        base_energy = dist_world * getattr(self.planner, 'baseline_energy_per_meter', 0.01)
        alt_change = goal_world[2] - node_world[2]
        
        p_energy = 0.0
        if alt_change > 0:
            total_mass = DRONE_MASS_KG + self.payload_kg
            joules = (total_mass * GRAVITY * alt_change) / ASCENT_EFFICIENCY
            p_energy = joules / 3600 # Convert Joules to Watt-hours

        return self.weight * (base_energy + p_energy)

class BalancedHeuristic(Heuristic):
    def __init__(self, planner: 'PathPlanner3D'):
        super().__init__(planner)
        self.time_h = TimeHeuristic(planner)
        self.energy_h = EnergyHeuristic(planner)
        self.max_time_est = 3600  # Default, will be updated dynamically
        self.max_energy_est = DRONE_BATTERY_WH # Default, will be updated dynamically

    def update_params(self, payload_kg: float, goal: GridCoord, time_weight: float = 0.5, **kwargs):
        super().update_params(payload_kg, goal, time_weight)
        self.time_h.update_params(payload_kg, goal, time_weight)
        self.energy_h.update_params(payload_kg, goal, time_weight)
        
        # Implement dynamic normalization based on the mission segment scale
        start_node = kwargs.get('start_node')
        if start_node:
            start_world = self.planner.coord_manager.grid_to_world(start_node)
            goal_world = self.planner.coord_manager.grid_to_world(goal)
            dist_world = np.linalg.norm(np.array(start_world) - np.array(goal_world))
            
            # Estimate max time with a 1.5x detour factor
            self.max_time_est = max(1.0, (dist_world / DRONE_SPEED_MPS) * 1.5)
            
            # Estimate max energy as a fraction of battery based on distance
            # A more robust estimation could be based on a simplified physics model
            est_joules = (DRONE_MASS_KG + payload_kg) * 9.81 * dist_world
            est_wh = (est_joules / 3600) * 1.5 # 1.5x factor for inefficiency and horizontal flight
            self.max_energy_est = max(1.0, est_wh)

    def calculate(self, node: GridCoord) -> float:
        # Normalize time and energy costs to make them comparable
        time_cost = self.time_h.calculate(node) / self.max_time_est
        energy_cost = self.energy_h.calculate(node) / self.max_energy_est
        
        combined_cost = (self.time_weight * time_cost + self.energy_weight * energy_cost)
        return combined_cost * self.weight