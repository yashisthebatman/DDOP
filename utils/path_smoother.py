import numpy as np
from scipy.interpolate import splev, splprep
import logging
from typing import List, Tuple, Dict

class PathSmoother:
    """Post-processes grid-based paths to create smoother B-spline curves."""
    
    def smooth_path(self, path: List[Tuple], env) -> List[Tuple]:
        """Generates a smooth B-spline path and discretizes it."""
        num_points_in_path = len(path)
        
        # FIX: The spline degree 'k' must be less than the number of points 'm'.
        # Dynamically adjust k for short paths to prevent scipy error.
        if num_points_in_path < 2:
            return path
        
        # k must be 1, 2, or 3. Max possible k is num_points - 1.
        spline_degree = min(num_points_in_path - 1, 3)

        if spline_degree < 1:
             return path # Cannot create a spline

        try:
            path_np = np.array(path).T
            # Use the calculated spline_degree
            tck, u = splprep(path_np, s=2.0, k=spline_degree)
            
            num_points_out = num_points_in_path * 5
            u_new = np.linspace(u.min(), u.max(), num_points_out)
            x_new, y_new, z_new = splev(u_new, tck, der=0)
            
            smoothed_path = list(zip(x_new, y_new, z_new))

            for i in range(len(smoothed_path) - 1):
                if env.is_line_obstructed(smoothed_path[i], smoothed_path[i+1]):
                    logging.warning("Path smoothing created a collision. Reverting to original path.")
                    return path
            
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
        if len(agent_ids) < 2:
            return True

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