import heapq
import logging
from typing import List, Tuple, Optional
from itertools import product
import numpy as np

from fleet.cbs_components import Agent, Constraint
from utils.coordinate_manager import CoordinateManager
from environment import Environment

GridPos = Tuple[int, int, int]

class TimeAwareAStar:
    """Low-level planner for finding a single agent's path on a discrete grid,
       respecting time-based constraints."""
       
    def __init__(self, env: Environment, coord_manager: CoordinateManager):
        self.env = env
        self.coord_manager = coord_manager
        self.last_agents: List[Agent] = []
        self.MOVES = [move for move in product([-1, 0, 1], repeat=3) if move != (0, 0, 0)]
        
    def find_path(self, agent: Agent, constraints: List[Constraint]) -> Optional[List[Tuple[GridPos, int]]]:
        self.last_agents.append(agent)
        start_node = (agent.start_pos, 0) # State is (position, time)
        
        # Priority queue stores (f_score, state)
        open_set = [(0, start_node)]
        came_from = {}
        
        # FIX: g_score must now store the *accumulated weighted cost*, not just time.
        # This is the critical bug fix.
        g_score = {start_node: 0.0}
        
        constraint_set = {(c.position, c.timestamp) for c in constraints}

        while open_set:
            _, current_state = heapq.heappop(open_set)
            current_pos, current_time = current_state

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

                move_distance = np.linalg.norm(np.array(move))
                edge_cost = (w_time * move_distance) + (w_risk * risk_cost * move_distance * 10)
                
                # g_score is the accumulated weighted cost from the start
                tentative_g_score = g_score[current_state] + edge_cost

                if neighbor_state not in g_score or tentative_g_score < g_score.get(neighbor_state, float('inf')):
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