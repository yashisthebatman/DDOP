# FILE: planners/cbsh_planner.py
import heapq
import logging
from typing import Dict, List, Optional, Tuple
from itertools import combinations
import numpy as np

from fleet.cbs_components import Agent, CTNode, Conflict, Constraint
from utils.coordinate_manager import CoordinateManager
from utils.path_smoother import PathSmoother
from environment import Environment
from utils.rrt_star_anytime import AnytimeRRTStar
from utils.path_timing_solver import PathTimingSolver
from utils.geometry import calculate_distance_3d


# FIX: The original time budget was too low for the RRT* planner to reliably
# find paths in the complex, obstacle-dense environment. Increasing it gives
# the low-level planner a higher chance of success.
LOW_LEVEL_TIME_BUDGET = 2.0
# Drones must be at least this far apart to be considered conflict-free
MIN_SEPARATION_METERS = 15.0

class CBSHPlanner:
    """
    Conflict-Based Search with Heuristics (CBSH).
    A high-level planner that uses a continuous-space, anytime low-level planner.
    """
    def __init__(self, environment: Environment, coord_manager: CoordinateManager):
        self.env = environment
        self.coord_manager = coord_manager
        self.smoother = PathSmoother()
        self.timing_solver = PathTimingSolver(coord_manager)
        self.agents_map: Dict[any, Agent] = {}

    def plan_fleet(self, agents: List[Agent]) -> Optional[Dict]:
        """
        Plans conflict-free paths for a list of agents.
        Returns a dictionary mapping agent_id to its timed, continuous-space path.
        """
        self.agents_map = {agent.id: agent for agent in agents}
        
        root = CTNode()
        for agent in agents:
            timed_path = self._find_path_for_agent(agent, [])
            if not timed_path:
                logging.error(f"Could not find initial path for agent {agent.id}. Problem is unsolvable.")
                return None
            root.solution[agent.id] = timed_path
        
        root.cost = self._calculate_solution_cost(root.solution)
        
        open_set = [root]

        while open_set:
            p_node = heapq.heappop(open_set)
            
            conflict = self._select_best_conflict(p_node.solution)
            if not conflict:
                logging.info(f"CBSH search complete. Solution found with cost {p_node.cost}.")
                return p_node.solution

            a1_id, a2_id = conflict.agent1_id, conflict.agent2_id
            
            for agent_id in [a1_id, a2_id]:
                agent_to_constrain = self.agents_map.get(agent_id)
                if not agent_to_constrain: continue

                new_constraints = p_node.constraints.copy()
                new_constraint = Constraint(agent_id, conflict.position, conflict.timestamp)
                new_constraints.add(new_constraint)

                agent_specific_constraints = [c for c in new_constraints if c.agent_id == agent_id]
                
                new_timed_path = self._find_path_for_agent(agent_to_constrain, agent_specific_constraints)

                if new_timed_path:
                    new_solution = p_node.solution.copy()
                    new_solution[agent_id] = new_timed_path
                    
                    child = CTNode(constraints=new_constraints, solution=new_solution)
                    child.cost = self._calculate_solution_cost(child.solution)
                    heapq.heappush(open_set, child)
                    
        logging.warning("CBSH search exhausted. No solution found.")
        return None

    def _find_path_for_agent(self, agent: Agent, constraints: List[Constraint]) -> Optional[List[Tuple[Tuple, int]]]:
        """
        Runs the two-stage low-level planning process.
        """
        rrt = AnytimeRRTStar(agent.start_pos, agent.goal_pos, self.env, self.coord_manager)
        geometric_path, _ = rrt.plan(time_budget_s=LOW_LEVEL_TIME_BUDGET)
        
        if not geometric_path:
            logging.warning(f"CBSH: RRT* failed for {agent.id}")
            return None
            
        timed_path = self.timing_solver.find_timing(geometric_path, constraints)
        
        if not timed_path:
            logging.warning(f"CBSH: Timing Solver failed for {agent.id} with {len(constraints)} constraints.")
            return None
            
        return timed_path

    def _calculate_solution_cost(self, solution: Dict) -> int:
        """Calculates the Sum of Individual Costs (SIC), where cost is travel time."""
        return sum(path[-1][1] for path in solution.values() if path)

    def _select_best_conflict(self, solution: Dict) -> Optional[Conflict]:
        """
        Finds all conflicts by checking for agents in the same grid cell at the same time,
        and selects the one that occurs earliest.
        """
        conflicts = []
        max_time = 0
        if solution.values():
             max_time = max((p[-1][1] for p in solution.values() if p), default=0)
        
        agent_ids = list(solution.keys())
        interpolated_paths = {agent_id: self._get_interpolated_path(path) for agent_id, path in solution.items()}

        for t in range(max_time + 1):
            positions_at_t = []
            for agent_id in agent_ids:
                if t < len(interpolated_paths[agent_id]) and interpolated_paths[agent_id][t] is not None:
                    positions_at_t.append((agent_id, interpolated_paths[agent_id][t]))

            for (id1, pos1), (id2, pos2) in combinations(positions_at_t, 2):
                # FIX: The original conflict detection used continuous distance (MIN_SEPARATION_METERS), 
                # but the constraints are discrete (grid-based). This mismatch caused an infinite 
                # loop where the planner would find a proximity conflict that its vertex-based 
                # constraints could not resolve.
                # This new logic detects conflicts only when agents are in the same grid cell,
                # which is consistent with the constraints the planner can create.
                grid_pos1 = self.coord_manager.world_to_local_grid(pos1)
                grid_pos2 = self.coord_manager.world_to_local_grid(pos2)
                
                if grid_pos1 and grid_pos1 == grid_pos2:
                    conflicts.append(Conflict(id1, id2, grid_pos1, t))

        return min(conflicts, key=lambda c: c.timestamp) if conflicts else None

    def _get_interpolated_path(self, timed_path: List[Tuple[Tuple, int]]) -> List[Tuple]:
        """Converts a timed waypoint path into a path with a position for every integer time step."""
        if not timed_path:
            return []
        
        max_time = timed_path[-1][1]
        full_path = [None] * (max_time + 1)
        
        for i in range(len(timed_path) - 1):
            p1, t1 = timed_path[i]
            p2, t2 = timed_path[i+1]
            
            full_path[t1] = p1
            
            segment_duration = t2 - t1
            if segment_duration > 0:
                for t_step in range(1, segment_duration):
                    progress = t_step / segment_duration
                    interp_pos = tuple(np.array(p1) + progress * (np.array(p2) - np.array(p1)))
                    full_path[t1 + t_step] = interp_pos
        
        if max_time < len(full_path):
            full_path[max_time] = timed_path[-1][0]
        return full_path