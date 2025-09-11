# FILE: planners/single_agent_planner.py
import logging
from typing import Tuple, List, Optional
import numpy as np

from environment import Environment
from ml_predictor.predictor import EnergyTimePredictor
from utils.coordinate_manager import CoordinateManager
from utils.geometry import point_in_aabb
from utils.rrt_star_anytime import AnytimeRRTStar # MODIFIED IMPORT

# Time budget for a quick tactical replan
TACTICAL_RRT_BUDGET_S = 0.2

class SingleAgentPlanner:
    """
    Handles tactical replanning for a single agent using a fast, localized RRT* search
    to create detours around new dynamic obstacles.
    """
    def __init__(self, env: Environment, predictor: EnergyTimePredictor, coord_manager: CoordinateManager):
        self.env = env
        self.predictor = predictor
        self.coord_manager = coord_manager

    def find_strategic_path_rrt(self, start_pos: Tuple, end_pos: Tuple) -> Tuple[Optional[List[Tuple]], str]:
        """Uses RRT* for long-range, obstacle-aware strategic planning."""
        if self.env.is_point_obstructed(start_pos): return None, "Start point is obstructed."
        if self.env.is_point_obstructed(end_pos): return None, "Destination point is obstructed."
        
        # Using the anytime version with a longer budget for strategic plans
        rrt_planner = AnytimeRRTStar(start_pos, end_pos, self.env, self.coord_manager)
        path, status = rrt_planner.plan(time_budget_s=1.0)
        
        return path, status

    def perform_hybrid_replan(self, current_pos: Tuple, stale_path: List[Tuple], new_obstacle_bounds: Tuple) -> Tuple[Optional[List[Tuple]], str]:
        """
        Performs a tactical replan using a time-limited RRT* to navigate around a new obstacle.
        """
        logging.info("--- Starting Single-Agent Tactical Replan (RRT*) ---")
        
        tactical_goal = None
        remaining_path_index = -1
        
        # Find the first waypoint on the old path that is clear of the new obstacle
        for i, waypoint in enumerate(stale_path):
            if i <= stale_path.index(current_pos): continue # Don't replan to the past
            if not point_in_aabb(waypoint, new_obstacle_bounds):
                tactical_goal = waypoint
                remaining_path_index = i
                logging.info(f"Safe tactical goal found at index {i}: {waypoint}")
                break
        
        if tactical_goal is None:
            logging.error("Drone is trapped. No safe waypoint found on the original path.")
            return None, "Fatal: Drone is trapped."
        
        # Use a fast, time-limited RRT* for the detour
        detour_planner = AnytimeRRTStar(current_pos, tactical_goal, self.env, self.coord_manager)
        detour_path, status = detour_planner.plan(time_budget_s=TACTICAL_RRT_BUDGET_S)

        if not detour_path:
            return None, f"Tactical RRT* escape failed: {status}"
        
        logging.info(f"Tactical detour found with {len(detour_path)} waypoints.")

        remaining_stale_path = stale_path[remaining_path_index:]
        # Splice the path: detour (without its last point) + rest of original path
        full_new_path = detour_path[:-1] + remaining_stale_path
        
        logging.info("--- Tactical Replan Successful ---")
        return full_new_path, "Hybrid replan successful."