# FILE: utils/path_smoother.py
import numpy as np
from scipy.interpolate import splev, splprep
import logging
from typing import List, Tuple, Dict

from config import PATH_SMOOTHING_OBSTACLE_CLEARANCE_METERS

class PathSmoother:
    """Post-processes grid-based paths to create smoother B-spline curves."""
    
    def smooth_path(self, path: List[Tuple], env, depth=0) -> List[Tuple]:
        """
        Generates a smooth B-spline path, validates it against obstacles,
        and uses recursion to smooth problematic segments.
        """
        if not path: return []
        
        # De-duplicate consecutive points that are too close for spline math.
        unique_path = [path[0]]
        for point in path[1:]:
            if np.linalg.norm(np.array(point) - np.array(unique_path[-1])) > 1e-6:
                unique_path.append(point)
        
        # FIX: If de-duplication results in a path that's too short for a spline,
        # return the de-duplicated path itself. This prevents collapsing a valid
        # short path (e.g., [A, B] where A is close to B) into an invalid one-point path.
        if len(unique_path) < 2:
            return unique_path # Return de-duplicated path

        # FIX: Removed the overly cautious check for `len < 4`. The spline
        # degree calculation correctly handles shorter paths (2 or 3 points),
        # allowing them to be densified into smoother lines.
        path = unique_path # Use the cleaned path for smoothing
        
        if depth > 2:
            logging.warning("Max smoothing recursion depth reached. Returning original path segment.")
            return path
        
        num_points_in_path = len(path)
        spline_degree = min(num_points_in_path - 1, 3)

        try:
            path_np = np.array(path).T
            tck, u = splprep(path_np, s=0, k=spline_degree)
            num_points_out = max(num_points_in_path * 5, 20)
            u_new = np.linspace(u.min(), u.max(), num_points_out)
            x_new, y_new, z_new = splev(u_new, tck, der=0)
            
            smoothed_path = list(zip(x_new, y_new, z_new))

            for i in range(len(smoothed_path) - 1):
                p1, p2 = smoothed_path[i], smoothed_path[i+1]
                num_interp_points = 5
                for j in range(num_interp_points + 1):
                    interp_point = tuple(np.array(p1) + (j / num_interp_points) * (np.array(p2) - np.array(p1)))
                    if env.is_point_obstructed(interp_point):
                        logging.warning(f"Smoothed path segment {i} collided with obstacle at depth {depth}. Subdividing.")
                        mid_index = num_points_in_path // 2
                        first_half = self.smooth_path(path[:mid_index+1], env, depth + 1)
                        second_half = self.smooth_path(path[mid_index:], env, depth + 1)
                        return first_half[:-1] + second_half

            return smoothed_path
        except Exception as e:
            logging.error(f"Failed to smooth path: {e}. Returning original path.")
            return path
            
    def validate_smoothed_solution(self, solution: Dict[str, List[Tuple]]) -> bool:
        """
        Performs a final check for dynamic collisions between smoothed paths.
        Assumes each drone moves one waypoint per time step.
        """
        agent_ids = list(solution.keys())
        if len(agent_ids) < 2: return True

        max_time = max(len(p) for p in solution.values())

        for t in range(max_time):
            positions_at_t = {}
            for agent_id in agent_ids:
                path = solution[agent_id]
                if t < len(path):
                    pos_tuple = tuple(np.round(path[t], 3))
                    if pos_tuple in positions_at_t:
                        logging.warning(f"Dynamic collision detected after smoothing at {pos_tuple}, t={t}")
                        return False
                    positions_at_t[pos_tuple] = agent_id
        return True