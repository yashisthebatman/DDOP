# ==============================================================================
# File: path_planner.py (Phase 3: Path Output Consistency)
# ==============================================================================

import logging
from typing import List, Optional, Tuple

import numpy as np
from utils.jump_point_search import JumpPointSearch
from utils.coordinate_manager import CoordinateManager
from utils.a_star import AStarSearch
from utils.geometry import line_box_intersection

def _round_tuple(t, ndigits=6):
    """Round a tuple of floats for output consistency."""
    return tuple(round(float(x), ndigits) for x in t)

class PathPlanner3D:
    def __init__(self, coord_manager: CoordinateManager, is_grid_obstructed, is_line_obstructed, heuristic):
        self.coord_manager = coord_manager
        self.is_grid_obstructed = is_grid_obstructed
        self.is_line_obstructed = is_line_obstructed
        self.heuristic = heuristic

    def plan_path(
        self,
        waypoints: List[Tuple[float, float, float]],
    ) -> Tuple[Optional[List[Tuple[float, float, float]]], str]:
        full_path = []
        for i in range(len(waypoints) - 1):
            start_world, end_world = waypoints[i], waypoints[i + 1]
            safe_start_grid = self.coord_manager.safe_grid_conversion(start_world)
            safe_end_grid = self.coord_manager.safe_grid_conversion(end_world)
            if safe_start_grid is None or safe_end_grid is None:
                error_msg = f"Planner: Grid conversion failed for segment {i}: {start_world} -> {end_world}"
                logging.error(error_msg)
                return None, error_msg

            if self.is_grid_obstructed(safe_start_grid):
                error_msg = f"Planner: Start grid cell obstructed for segment {i}: {safe_start_grid}"
                logging.error(error_msg)
                return None, error_msg
            if self.is_grid_obstructed(safe_end_grid):
                error_msg = f"Planner: End grid cell obstructed for segment {i}: {safe_end_grid}"
                logging.error(error_msg)
                return None, error_msg

            jps = JumpPointSearch(
                start=safe_start_grid,
                goal=safe_end_grid,
                is_obstructed_func=self.is_grid_obstructed,
                heuristic=self.heuristic,
                coord_manager=self.coord_manager,
            )
            path_segment_grid = jps.search()
            if path_segment_grid is None:
                logging.warning(f"Planner: JPS failed on segment {i}. Falling back to A*.")
                a_star = AStarSearch(
                    safe_start_grid,
                    safe_end_grid,
                    self.is_grid_obstructed,
                    self.heuristic,
                    self.coord_manager,
                )
                path_segment_grid = a_star.search()
                if path_segment_grid is None:
                    error_msg = f"Planner: Fatal Error: Both JPS and A* failed on segment {i}."
                    logging.error(error_msg)
                    return None, error_msg
            # Convert to world coordinates and round for output consistency
            path_segment_world = [_round_tuple(self.coord_manager.grid_to_world(p)) for p in path_segment_grid]
            # Remove duplicate at joins
            if full_path and path_segment_world:
                if full_path[-1] == path_segment_world[0]:
                    path_segment_world = path_segment_world[1:]
            full_path.extend(path_segment_world)
        # Ensure output starts at requested start and ends at requested goal
        if full_path:
            if _round_tuple(waypoints[0]) != full_path[0]:
                full_path = [_round_tuple(waypoints[0])] + full_path
            if _round_tuple(waypoints[-1]) != full_path[-1]:
                full_path.append(_round_tuple(waypoints[-1]))
        # Remove any accidental duplicates
        consistent_path = []
        for pt in full_path:
            if not consistent_path or pt != consistent_path[-1]:
                consistent_path.append(pt)
        simplified_path = self._simplify_path(consistent_path)
        logging.info(f"Planner: Path simplified from {len(full_path)} to {len(simplified_path)} waypoints.")
        return simplified_path, "Path found successfully."

    def _simplify_path(self, path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        if not path or len(path) < 3:
            return path
        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            prev, curr, next_ = path[i - 1], path[i], path[i + 1]
            if self._is_colinear(prev, curr, next_):
                continue
            simplified.append(curr)
        simplified.append(path[-1])
        return simplified

    def _is_colinear(self, p1, p2, p3, tol=1e-6):
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p2)
        cross = np.cross(v1, v2)
        return np.linalg.norm(cross) < tol
