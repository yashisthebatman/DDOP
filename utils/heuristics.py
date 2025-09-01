# utils/heuristics.py (Full Updated Code)
import numpy as np
from typing import Tuple, Optional
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
        Calculates the actual cost of moving between two adjacent nodes using the
        pre-computed cost map. THIS IS NOW EXTREMELY FAST.
        """
        dist = np.linalg.norm(np.array(n1) - np.array(n2))
        
        # Fetch the pre-computed cost multiplier for the destination node.
        # If a cell is not in the map, it's open air with a default cost.
        cost_multiplier = self.planner.cost_map.get(n2, config.DEFAULT_CELL_COST)
        
        # If the destination is an obstacle, the cost is infinite.
        if cost_multiplier == float('inf'):
            return float('inf')
        
        # The cost is simply the geometric distance scaled by the pre-computed environmental cost.
        return dist * cost_multiplier

class TimeHeuristic(Heuristic):
    # ... `calculate` method is unchanged ...
    def calculate(self, node: GridCoord) -> float:
        node_world=self.planner._grid_to_world(node);goal_world=self.planner._grid_to_world(self.goal)
        dist=np.linalg.norm(np.array(goal_world)-np.array(node_world))
        if dist<1e-6:return 0.0
        avg_wind=self.planner.env.weather.get_wind_at_location(node_world[0],node_world[1]);flight_vec=np.array(goal_world)-np.array(node_world)
        time_impact,_=calculate_wind_effect(flight_vec,avg_wind,config.DRONE_SPEED_MPS);est_cost=(dist*self.planner.baseline_time_per_meter)*time_impact
        return self.weight*est_cost

class EnergyHeuristic(Heuristic):
    # ... `calculate` method is unchanged ...
    def calculate(self, node: GridCoord) -> float:
        node_world=self.planner._grid_to_world(node);goal_world=self.planner._grid_to_world(self.goal)
        dist=np.linalg.norm(np.array(goal_world)-np.array(node_world))
        avg_wind=self.planner.env.weather.get_wind_at_location(node_world[0],node_world[1]);flight_vec=np.array(goal_world)-np.array(node_world)
        _,energy_impact=calculate_wind_effect(flight_vec,avg_wind,config.DRONE_SPEED_MPS);h_energy=dist*self.planner.baseline_energy_per_meter*energy_impact
        alt_change=goal_world[2]-node_world[2];p_energy=0.0
        if alt_change>0:
            total_mass=config.DRONE_MASS_KG+self.payload_kg;joules=(total_mass*config.GRAVITY*alt_change)/config.ASCENT_EFFICIENCY
            p_energy=joules/3600
        return self.weight*(h_energy+p_energy)

class BalancedHeuristic(Heuristic):
    # ... `calculate` method is unchanged ...
    def __init__(self, planner: PathPlanner, payload_kg: float, goal: GridCoord, time_weight: float = 0.5):
        super().__init__(planner, payload_kg, goal, time_weight)
        self.time_h=TimeHeuristic(planner,payload_kg,goal,time_weight);self.energy_h=EnergyHeuristic(planner,payload_kg,goal,time_weight)
        start_w,goal_w=planner._grid_to_world(goal),planner._grid_to_world(goal);dist=np.linalg.norm(np.array(start_w)-np.array(goal_w))
        self.max_time_est=(dist/config.DRONE_SPEED_MPS)*2+1;self.max_energy_est=(dist*planner.baseline_energy_per_meter)*2+1
    def calculate(self, node: GridCoord) -> float:
        time_c=self.time_h.calculate(node)/self.max_time_est;energy_c=self.energy_h.calculate(node)/self.max_energy_est
        return(self.time_weight*time_c)+(self.energy_weight*energy_c)