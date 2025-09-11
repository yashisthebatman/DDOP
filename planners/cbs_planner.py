import heapq
import logging
from itertools import combinations
from typing import Dict, List, Optional

from fleet.cbs_components import Agent, CTNode, Conflict, Constraint
from utils.time_aware_a_star import TimeAwareAStar
from utils.coordinate_manager import CoordinateManager
from utils.path_smoother import PathSmoother
from environment import Environment

class CBSPlanner:
    """High-level Conflict-Based Search (CBS) planner."""
    def __init__(self, environment: Environment, coord_manager: CoordinateManager):
        self.env = environment
        self.coord_manager = coord_manager
        self.low_level_planner = TimeAwareAStar(environment, coord_manager)
        self.smoother = PathSmoother()

    def plan_fleet(self, agents: List[Agent]) -> Optional[Dict]:
        """
        Plans conflict-free paths for a list of agents.
        Returns a dictionary mapping agent_id to its path, or None if unsolvable.
        """
        root = CTNode()
        for agent in agents:
            path = self.low_level_planner.find_path(agent, [])
            if not path:
                logging.error(f"Could not find initial path for agent {agent.id}. Problem is unsolvable.")
                return None
            root.solution[agent.id] = path
        
        root.cost = self._calculate_solution_cost(root.solution)
        
        open_set = [root]

        while open_set:
            p_node = heapq.heappop(open_set)
            
            first_conflict = self._find_first_conflict(p_node.solution)
            if not first_conflict:
                logging.info(f"CBS search complete. Solution found with cost {p_node.cost}.")
                
                # Final validation and smoothing
                final_solution = {}
                for agent_id, path in p_node.solution.items():
                    world_path = [self.coord_manager.local_grid_to_world(pos_time[0]) for pos_time in path]
                    smoothed_path = self.smoother.smooth_path(world_path, self.env)
                    final_solution[agent_id] = smoothed_path
                
                # Re-check for collisions after smoothing
                if not self.smoother.validate_smoothed_solution(final_solution):
                    logging.warning("Path smoothing introduced a conflict! Returning original grid path.")
                    # Fallback to non-smoothed path if smoothing fails
                    return {agent_id: [self.coord_manager.local_grid_to_world(pt[0]) for pt in path] for agent_id, path in p_node.solution.items()}

                return final_solution

            a1_id, a2_id = first_conflict.agent1_id, first_conflict.agent2_id
            
            for agent_id in [a1_id, a2_id]:
                child_node = self._create_child_node(p_node, agent_id, first_conflict)
                if child_node:
                    heapq.heappush(open_set, child_node)
                    
        logging.warning("CBS search exhausted. No solution found.")
        return None
    
    def _create_child_node(self, parent: CTNode, agent_to_constrain: int, conflict: Conflict) -> Optional[CTNode]:
        new_constraints = parent.constraints.copy()
        new_constraint = Constraint(agent_to_constrain, conflict.position, conflict.timestamp)
        new_constraints.add(new_constraint)

        # Create a list of all constraints relevant to this specific agent
        agent_specific_constraints = [c for c in new_constraints if c.agent_id == agent_to_constrain]

        # Find the original agent object to pass to the low-level planner
        original_agent = next((agent for agent in self.low_level_planner.last_agents if agent.id == agent_to_constrain), None)
        if not original_agent: return None

        new_path = self.low_level_planner.find_path(original_agent, agent_specific_constraints)

        if not new_path:
            return None # This branch is invalid

        new_solution = parent.solution.copy()
        new_solution[agent_to_constrain] = new_path
        
        child = CTNode(constraints=new_constraints, solution=new_solution)
        child.cost = self._calculate_solution_cost(child.solution)
        return child

    def _calculate_solution_cost(self, solution: Dict) -> int:
        """Calculates the Sum of Individual Costs (SIC)."""
        return sum(len(path) for path in solution.values())

    def _find_first_conflict(self, solution: Dict) -> Optional[Conflict]:
        """Finds the first vertex or edge conflict in the solution."""
        max_time = max(len(p) for p in solution.values()) if solution else 0
        
        for t in range(max_time):
            # Vertex conflict check
            positions_at_t = {}
            for agent_id, path in solution.items():
                if t < len(path):
                    pos = path[t][0]
                    if pos in positions_at_t:
                        # Conflict found
                        other_agent_id = positions_at_t[pos]
                        return Conflict(agent_id, other_agent_id, pos, t)
                    positions_at_t[pos] = agent_id

        # Edge conflicts (swapping) can be added here if needed
        return None