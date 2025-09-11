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
from config import DRONE_SPEED_MPS
from utils.a_star import AStarPlanner

LOW_LEVEL_TIME_BUDGET = 2.0
MIN_SEPARATION_METERS = 15.0

class CBSHPlanner:
    def __init__(self, environment: Environment, coord_manager: CoordinateManager):
        self.env = environment
        self.coord_manager = coord_manager
        self.smoother = PathSmoother()
        self.timing_solver = PathTimingSolver(coord_manager)
        self.agents_map: Dict[any, Agent] = {}
        self.a_star_planner = AStarPlanner()
        self.planning_grid = self.env.create_planning_grid()

    def plan_fleet(self, agents: List[Agent]) -> Optional[Dict]:
        self.agents_map = {agent.id: agent for agent in agents}
        root = CTNode()
        for agent in agents:
            timed_path = self._find_path_for_agent(agent, [])
            if not timed_path: return None
            root.solution[agent.id] = timed_path
        root.cost = self._calculate_solution_cost(root.solution)
        open_set = [root]
        while open_set:
            p_node = heapq.heappop(open_set)
            conflict = self._select_best_conflict(p_node.solution)
            if not conflict:
                logging.info(f"CBSH search complete. Solution found with cost {p_node.cost}.")
                return p_node.solution
            for agent_id_to_constrain in [conflict.agent1_id, conflict.agent2_id]:
                new_constraints = p_node.constraints.copy()
                constraint_pos = conflict.agent1_pos if agent_id_to_constrain == conflict.agent1_id else conflict.agent2_pos
                new_constraint = Constraint(agent_id_to_constrain, constraint_pos, conflict.timestamp)
                new_constraints.add(new_constraint)
                agent_specific_constraints = [c for c in new_constraints if c.agent_id == agent_id_to_constrain]
                agent_to_replan = self.agents_map[agent_id_to_constrain]
                new_timed_path = self._find_path_for_agent(agent_to_replan, agent_specific_constraints)
                if new_timed_path:
                    new_solution = p_node.solution.copy()
                    new_solution[agent_id_to_constrain] = new_timed_path
                    child = CTNode(constraints=new_constraints, solution=new_solution)
                    child.cost = self._calculate_solution_cost(child.solution)
                    heapq.heappush(open_set, child)
        logging.warning("CBSH search exhausted. No solution found.")
        return None

    def _find_path_for_agent(self, agent: Agent, constraints: List[Constraint]) -> Optional[List[Tuple[Tuple, int]]]:
        if self.env.is_point_obstructed(agent.start_pos):
            logging.error(f"Could not plan for agent {agent.id}: Start point is obstructed.")
            return None
        if self.env.is_point_obstructed(agent.goal_pos):
            logging.error(f"Could not plan for agent {agent.id}: Goal point is obstructed.")
            return None
            
        start_m = self.coord_manager.world_to_meters(agent.start_pos)
        goal_m = self.coord_manager.world_to_meters(agent.goal_pos)
        start_grid, goal_grid = self.coord_manager.meters_to_grid(start_m), self.coord_manager.meters_to_grid(goal_m)
        if not start_grid or not goal_grid:
            logging.error(f"Agent {agent.id} start/goal is outside the planning grid.")
            return None
        
        strategic_grid_path = self.a_star_planner.find_path(self.planning_grid, start_grid, goal_grid)
        if not strategic_grid_path:
            logging.warning(f"CBSH: A* failed to find a strategic path for {agent.id}.")
            return None
        
        strategic_waypoints_world = [agent.start_pos]
        for grid_pos in strategic_grid_path[1:-1:5]: # Simplify path
             strategic_waypoints_world.append(self.coord_manager.meters_to_world(self.coord_manager.grid_to_meters(grid_pos)))
        strategic_waypoints_world.append(agent.goal_pos)
        
        full_geometric_path = [strategic_waypoints_world[0]]
        for i in range(len(strategic_waypoints_world) - 1):
            rrt = AnytimeRRTStar(strategic_waypoints_world[i], strategic_waypoints_world[i+1], self.env, self.coord_manager)
            segment_path, status = rrt.plan(time_budget_s=0.2)
            if not segment_path:
                logging.warning(f"CBSH: RRT* failed on segment {i} for {agent.id}. Status: {status}")
                return None
            full_geometric_path.extend(segment_path[1:])
        
        timed_path = self.timing_solver.find_timing(full_geometric_path, constraints)
        if not timed_path:
            logging.warning(f"CBSH: Timing Solver failed for {agent.id} with {len(constraints)} constraints.")
            return None
        return timed_path

    def _calculate_solution_cost(self, solution: Dict) -> float:
        total_cost = 0
        for agent_id, path in solution.items():
            if not path: continue
            agent = self.agents_map[agent_id]
            actual_travel_time = path[-1][1]
            start_m, goal_m = self.coord_manager.world_to_meters(agent.start_pos), self.coord_manager.world_to_meters(agent.goal_pos)
            dist_m = calculate_distance_3d(start_m, goal_m)
            heuristic_time = dist_m / DRONE_SPEED_MPS if DRONE_SPEED_MPS > 1e-6 else float('inf')
            normalized_cost = (0.7 * actual_travel_time) + (0.3 * heuristic_time)
            total_cost += normalized_cost
        return total_cost

    def _select_best_conflict(self, solution: Dict) -> Optional[Conflict]:
        # ... (unchanged)
        conflicts = []
        if not any(solution.values()): return None
        max_time = max((p[-1][1] for p in solution.values() if p), default=0)
        agent_ids = list(solution.keys())
        interpolated_paths = {agent_id: self._get_interpolated_path(path) for agent_id, path in solution.items()}
        for t in range(max_time + 1):
            positions_at_t = []
            for agent_id in agent_ids:
                if t < len(interpolated_paths[agent_id]) and interpolated_paths[agent_id][t] is not None:
                    positions_at_t.append((agent_id, interpolated_paths[agent_id][t]))
            for (id1, pos1), (id2, pos2) in combinations(positions_at_t, 2):
                pos1_m, pos2_m = self.coord_manager.world_to_meters(pos1), self.coord_manager.world_to_meters(pos2)
                if calculate_distance_3d(pos1_m, pos2_m) < MIN_SEPARATION_METERS:
                    grid_pos1, grid_pos2 = self.coord_manager.meters_to_grid(pos1_m), self.coord_manager.meters_to_grid(pos2_m)
                    if grid_pos1 and grid_pos2:
                         conflicts.append(Conflict(id1, id2, grid_pos1, grid_pos2, t))
        return min(conflicts, key=lambda c: c.timestamp) if conflicts else None

    def _get_interpolated_path(self, timed_path: List[Tuple[Tuple, int]]) -> List[Tuple]:
        # ... (unchanged)
        if not timed_path or not timed_path[-1]: return []
        max_time = timed_path[-1][1]
        full_path = [None] * (max_time + 1)
        if timed_path: full_path[0] = timed_path[0][0]
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
        if max_time < len(full_path): full_path[max_time] = timed_path[-1][0]
        return full_path