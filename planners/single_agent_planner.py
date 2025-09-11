import logging
from typing import Tuple, List, Optional
import numpy as np

from environment import Environment
from ml_predictor.predictor import EnergyTimePredictor
from utils.coordinate_manager import CoordinateManager
from utils.d_star_lite import DStarLite
from utils.geometry import point_in_aabb, calculate_distance_3d
from utils.heuristics import HeuristicProvider
from utils.rrt_star import RRTStar

class SingleAgentPlanner:
    """
    Handles path planning for a single agent, primarily for emergency replanning
    (e.g., Return-To-Hub) or when operating outside a fleet context.
    This class combines a strategic RRT* planner with a tactical D* Lite planner.
    """
    def __init__(self, env: Environment, predictor: EnergyTimePredictor, coord_manager: CoordinateManager):
        self.env = env
        self.predictor = predictor
        self.coord_manager = coord_manager
        self.heuristics = HeuristicProvider(self.coord_manager)

    def find_strategic_path_rrt(self, start_pos: Tuple, end_pos: Tuple) -> Tuple[Optional[List[Tuple]], str]:
        """Uses RRT* for long-range, obstacle-aware strategic planning."""
        if self.env.is_point_obstructed(start_pos): return None, "Start point is obstructed."
        if self.env.is_point_obstructed(end_pos): return None, "Destination point is obstructed."
        
        rrt_planner = RRTStar(start_pos, end_pos, self.env, self.coord_manager)
        path, status = rrt_planner.plan()
        
        return path, status if path else (None, status)

    def perform_hybrid_replan(self, current_pos: Tuple, stale_path: List[Tuple], new_obstacle_bounds: Tuple) -> Tuple[Optional[List[Tuple]], str]:
        """Performs a tactical replan using D* Lite to navigate around a new obstacle."""
        logging.info("--- Starting Single-Agent Hybrid Replan (D* Lite) ---")
        
        tactical_goal = None
        remaining_path_index = -1
        
        for i, waypoint in enumerate(stale_path[1:], start=1):
            if not point_in_aabb(waypoint, new_obstacle_bounds):
                tactical_goal = waypoint
                remaining_path_index = i
                logging.info(f"Safe tactical goal found at index {i}: {waypoint}")
                break
        
        if tactical_goal is None:
            logging.error("Drone is trapped. No safe waypoint found on the original path.")
            return None, "Fatal: Drone is trapped."
        
        # --- FIX: Moved grid sizing logic here from the tactical planner ---
        # This allows the orchestrator to set up the environment correctly.
        all_points_x = [current_pos[0], tactical_goal[0], new_obstacle_bounds[0], new_obstacle_bounds[3]]
        all_points_y = [current_pos[1], tactical_goal[1], new_obstacle_bounds[1], new_obstacle_bounds[4]]
        
        grid_center_world = (np.mean(all_points_x), np.mean(all_points_y), np.mean([current_pos[2], tactical_goal[2]]))
        grid_size_m = max(calculate_distance_3d(current_pos, tactical_goal) * 1.5, 500)
        self.coord_manager.set_local_grid_origin(grid_center_world, grid_size_m)
        
        detour_path, status = self._find_tactical_detour(current_pos, tactical_goal, new_obstacle_bounds)
        
        if not detour_path:
            return None, f"Tactical D* Lite escape failed: {status}"
        
        logging.info(f"D* Lite detour found with {len(detour_path)} waypoints.")

        remaining_stale_path = stale_path[remaining_path_index:]
        full_new_path = detour_path[:-1] + remaining_stale_path
        
        logging.info("--- Hybrid Replan Successful ---")
        return full_new_path, "Hybrid replan successful."

    def _find_tactical_detour(self, start_pos, end_pos, new_obstacle_bounds):
        # FIX: Grid sizing logic was removed from here.
        grid_start = self.coord_manager.world_to_local_grid(start_pos)
        grid_goal = self.coord_manager.world_to_local_grid(end_pos)
        
        if not grid_start: return None, f"Tactical Start {start_pos} is outside the local grid."
        if not grid_goal: return None, f"Tactical Goal {end_pos} is outside the local grid."

        cost_map = {}
        min_g = self.coord_manager.world_to_local_grid((new_obstacle_bounds[0], new_obstacle_bounds[1], new_obstacle_bounds[2]))
        max_g = self.coord_manager.world_to_local_grid((new_obstacle_bounds[3], new_obstacle_bounds[4], new_obstacle_bounds[5]))
        
        if min_g and max_g:
            for x in range(min(min_g[0], max_g[0]), max(min_g[0], max_g[0]) + 1):
                for y in range(min(min_g[1], max_g[1]), max(min_g[1], max_g[1]) + 1):
                    for z in range(min(min_g[2], max_g[2]), max(min_g[2], max_g[2]) + 1):
                        grid_cell = (x, y, z)
                        if self.coord_manager.is_valid_local_grid_pos(grid_cell):
                             cost_map[grid_cell] = float('inf')
        
        dstar = DStarLite(start=grid_start, goal=grid_goal, cost_map=cost_map, heuristic_provider=self.heuristics, coord_manager=self.coord_manager)
        dstar.compute_shortest_path()
        path_grid = dstar.get_path()
        
        if path_grid:
            path_world = [self.coord_manager.local_grid_to_world(p) for p in path_grid if p is not None]
            return path_world, "Path found."
        
        return None, "D* Lite could not find a tactical path."