import logging
from typing import List, Optional, Tuple
import numpy as np

from utils.rrt_star import RRTStar
from utils.d_star_lite import DStarLite
from utils.coordinate_manager import CoordinateManager
from utils.heuristics import HeuristicProvider
from utils.geometry import point_in_aabb
from environment import Environment
from ml_predictor.predictor import EnergyTimePredictor

class PathPlanner3D:
    def __init__(self, env: Environment, predictor: EnergyTimePredictor):
        self.env = env
        self.predictor = predictor
        self.coord_manager = CoordinateManager()
        self.heuristics = HeuristicProvider(self.coord_manager)

    def find_path(self, start_pos: Tuple, end_pos: Tuple, payload_kg: float, mode: str, time_weight: float = 0.5) -> Tuple[Optional[List[Tuple]], str]:
        if self.env.is_point_obstructed(start_pos): return None, "Start point is obstructed."
        if self.env.is_point_obstructed(end_pos): return None, "Destination point is obstructed."
        
        rrt_planner = RRTStar(start_pos, end_pos, self.env, self.coord_manager)
        path, status = rrt_planner.plan()
        
        if not path:
            return None, status
        
        return path[1:], status

    def perform_hybrid_replan(self, current_pos: Tuple, stale_path: List[Tuple], new_obstacle_bounds: Tuple) -> Tuple[Optional[List[Tuple]], str]:
        logging.info("--- Starting Hybrid Replan (RRT*/D* Lite) ---")
        
        tactical_goal = None
        remaining_path_index = -1
        for i, waypoint in enumerate(stale_path):
            if not point_in_aabb(waypoint, new_obstacle_bounds):
                tactical_goal = waypoint
                remaining_path_index = i
                break
        
        if tactical_goal is None:
            return None, "Fatal: Drone is trapped. No safe waypoint found on the original path."
        
        logging.info(f"Tactical goal identified on old path: {tactical_goal}")

        detour_path, status = self._find_tactical_detour(current_pos, tactical_goal, new_obstacle_bounds)
        
        if not detour_path:
            return None, f"Tactical D* Lite escape failed: {status}"
        
        logging.info(f"D* Lite detour found with {len(detour_path)} waypoints.")

        # FIX: Correctly stitch the path together.
        # The remainder of the old path starts *at* the tactical goal.
        remaining_stale_path = stale_path[remaining_path_index:]
        
        # The new path is the detour (which ends at the tactical goal), minus its last
        # element to avoid duplication, plus the rest of the old path.
        full_new_path = detour_path[:-1] + remaining_stale_path
        
        logging.info("--- Hybrid Replan Successful ---")
        return full_new_path, "Hybrid replan successful."

    def _find_tactical_detour(self, start_pos, end_pos, new_obstacle_bounds):
        self.coord_manager.set_local_grid_origin(start_pos)
        grid_start = self.coord_manager.world_to_local_grid(start_pos)
        grid_goal = self.coord_manager.world_to_local_grid(end_pos)
        
        if not grid_start or not grid_goal:
            return None, "Start or end of tactical path is outside the local grid."

        cost_map = {}
        min_g = self.coord_manager.world_to_local_grid((new_obstacle_bounds[0], new_obstacle_bounds[1], new_obstacle_bounds[2]))
        max_g = self.coord_manager.world_to_local_grid((new_obstacle_bounds[3], new_obstacle_bounds[4], new_obstacle_bounds[5]))
        if min_g and max_g:
            for x in range(min(min_g[0], max_g[0]), max(min_g[0], max_g[0]) + 1):
                for y in range(min(min_g[1], max_g[1]), max(min_g[1], max_g[1]) + 1):
                    for z in range(min(min_g[2], max_g[2]), max(min_g[2], max_g[2]) + 1):
                        if self.coord_manager.is_valid_local_grid_pos((x,y,z)): cost_map[(x,y,z)] = float('inf')
        
        dstar = DStarLite(start=grid_start, goal=grid_goal, cost_map=cost_map, heuristic_provider=self.heuristics, coord_manager=self.coord_manager, mode='time')
        dstar.compute_shortest_path()
        path_grid = dstar.get_path()
        
        if path_grid:
            path_world = [self.coord_manager.local_grid_to_world(p) for p in path_grid if p is not None]
            return path_world, "Path found."
        
        return None, "D* Lite could not find a tactical path."