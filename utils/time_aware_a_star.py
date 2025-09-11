import heapq
import logging
from typing import List, Tuple, Optional
from itertools import product
import numpy as np

from fleet.cbs_components import Agent, Constraint
from utils.coordinate_manager import CoordinateManager
from environment import Environment

GridPos = Tuple[int, int, int]
MAX_TIME_STEPS = 150 # Safeguard against infinite searches in impossible scenarios

class TimeAwareAStar:
    """Low-level planner for finding a single agent's path on a discrete grid,
       respecting time-based constraints."""
       
    def __init__(self, env: Environment, coord_manager: CoordinateManager):
        self.env = env
        self.coord_manager = coord_manager
        self.last_agents: List[Agent] = []
        
        # PERFORMANCE OPTIMIZATION: Pre-calculate move costs to avoid repeated np.linalg.norm calls.
        self.MOVES = [move for move in product([-1, 0, 1], repeat=3)]
        self.MOVE_COSTS = {move: np.linalg.norm(np.array(move)) for move in self.MOVES}
        
    def find_path(self, agent: Agent, constraints: List[Constraint]) -> Optional[List[Tuple[GridPos, int]]]:
        self.last_agents.append(agent)
        start_node = (agent.start_pos, 0) # State is (position, time)
        
        open_set = [(0, start_node)]
        came_from = {}
        
        g_score = {start_node: 0.0}
        
        constraint_set = {(c.position, c.timestamp) for c in constraints}

        while open_set:
            _, current_state = heapq.heappop(open_set)
            current_pos, current_time = current_state
            
            if current_time > MAX_TIME_STEPS:
                continue

            if current_pos == agent.goal_pos:
                return self._reconstruct_path(came_from, current_state)

            for move in self.MOVES:
                neighbor_pos = tuple(np.add(current_pos, move))

                if not self.coord_manager.is_valid_local_grid_pos(neighbor_pos):
                    continue
                
                neighbor_time = current_time + 1
                neighbor_state = (neighbor_pos, neighbor_time)
                
                if neighbor_state in constraint_set:
                    continue

                w_time = agent.config.get('w_time', 1.0)
                w_risk = agent.config.get('w_risk', 0.0)
                
                world_pos = self.coord_manager.local_grid_to_world(neighbor_pos)
                risk_cost = self.env.risk_map.get_risk(world_pos) if world_pos else 0

                # Use fast pre-calculated move cost
                move_distance = self.MOVE_COSTS[move]
                
                wait_cost = 0.1 if move_distance == 0 else 0
                edge_cost = (w_time * move_distance) + wait_cost + (w_risk * risk_cost * 10)
                
                tentative_g_score = g_score.get(current_state, float('inf')) + edge_cost

                if tentative_g_score < g_score.get(neighbor_state, float('inf')):
                    g_score[neighbor_state] = tentative_g_score
                    heuristic_cost = np.linalg.norm(np.array(agent.goal_pos) - np.array(neighbor_pos))
                    f_score = tentative_g_score + (w_time * heuristic_cost)
                    
                    came_from[neighbor_state] = current_state
                    heapq.heappush(open_set, (f_score, neighbor_state))
                    
        logging.warning(f"Time-Aware A* failed to find a path for agent {agent.id}")
        return None

    def _reconstruct_path(self, came_from, current) -> List[Tuple[GridPos, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]