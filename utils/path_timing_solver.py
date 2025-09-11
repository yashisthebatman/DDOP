# FILE: utils/path_timing_solver.py
import heapq
import logging
from typing import List, Tuple, Optional, Dict
import numpy as np

from fleet.cbs_components import Constraint
from utils.coordinate_manager import CoordinateManager
from utils.geometry import calculate_distance_3d
from config import DRONE_SPEED_MPS

# Type aliases for clarity
WorldPosition = Tuple[float, float, float]
State = Tuple[int, int] # (waypoint_index, time)

MAX_WAIT_TIME = 10 # Max number of timesteps an agent can wait at one waypoint
MAX_TIME_STEPS = 150 # Safeguard against infinite searches

class PathTimingSolver:
    """
    Finds a valid timing schedule for a given geometric path, respecting
    space-time constraints from a CBS planner.
    """
    def __init__(self, coord_manager: CoordinateManager):
        self.coord_manager = coord_manager

    def _get_consecutive_waits(self, state: State, came_from: Dict[State, State]) -> int:
        """Helper to trace back and count consecutive wait states."""
        wait_count = 0
        current_s = state
        current_idx = state[0]
        
        while current_s in came_from:
            parent_s = came_from[current_s]
            parent_idx = parent_s[0]
            if parent_idx == current_idx:
                wait_count += 1
                current_s = parent_s
            else:
                # The parent was at a different waypoint, so this state was the result of a move.
                break
        return wait_count

    def find_timing(self, geometric_path: List[WorldPosition], constraints: List[Constraint]) -> Optional[List[Tuple[WorldPosition, int]]]:
        """
        Performs an A* search over the state space (waypoint_index, time) to find
        a conflict-free traversal schedule.
        """
        if not geometric_path:
            return []

        start_state: State = (0, 0) # (waypoint_idx, time)
        goal_waypoint_idx = len(geometric_path) - 1

        # Convert world-space constraints to grid-space for efficient lookup
        constraint_set = {(c.position, c.timestamp) for c in constraints}

        open_set = [(0, start_state)] # (f_score, state)
        came_from: Dict[State, State] = {}
        g_score: Dict[State, int] = {start_state: 0} # Time is the g_score

        while open_set:
            _, current_state = heapq.heappop(open_set)
            current_idx, current_time = current_state

            if current_idx == goal_waypoint_idx:
                return self._reconstruct_timed_path(came_from, current_state, geometric_path)

            if current_time > MAX_TIME_STEPS:
                continue

            # --- Explore Neighbors (Actions: Move or Wait) ---

            # 1. Action: Move to the next waypoint
            if current_idx < goal_waypoint_idx:
                p1_world = geometric_path[current_idx]
                p2_world = geometric_path[current_idx + 1]
                
                p1_meters = self.coord_manager.world_to_local_meters(p1_world)
                p2_meters = self.coord_manager.world_to_local_meters(p2_world)
                dist_m = calculate_distance_3d(p1_meters, p2_meters)

                # Ensure we don't divide by zero for very short segments
                if DRONE_SPEED_MPS > 1e-6:
                    time_to_travel = int(np.ceil(dist_m / DRONE_SPEED_MPS))
                else:
                    time_to_travel = float('inf')
                
                # A move always takes at least 1 time step
                neighbor_time = current_time + max(1, time_to_travel)
                neighbor_idx = current_idx + 1
                neighbor_state: State = (neighbor_idx, neighbor_time)
                
                if self._is_valid_state(neighbor_state, geometric_path, constraint_set):
                    if neighbor_time < g_score.get(neighbor_state, float('inf')):
                        came_from[neighbor_state] = current_state
                        g_score[neighbor_state] = neighbor_time
                        h_score = self._heuristic(neighbor_idx, goal_waypoint_idx, geometric_path)
                        f_score = neighbor_time + h_score
                        heapq.heappush(open_set, (f_score, neighbor_state))

            # 2. Action: Wait at the current waypoint
            # FIX: The original logic for checking wait time was flawed.
            # This now correctly counts the total number of consecutive waits.
            consecutive_waits = self._get_consecutive_waits(current_state, came_from)
            if consecutive_waits < MAX_WAIT_TIME:
                neighbor_time = current_time + 1
                neighbor_idx = current_idx
                neighbor_state: State = (neighbor_idx, neighbor_time)
                
                if self._is_valid_state(neighbor_state, geometric_path, constraint_set):
                     if neighbor_time < g_score.get(neighbor_state, float('inf')):
                        came_from[neighbor_state] = current_state
                        g_score[neighbor_state] = neighbor_time
                        h_score = self._heuristic(neighbor_idx, goal_waypoint_idx, geometric_path)
                        f_score = neighbor_time + h_score
                        heapq.heappush(open_set, (f_score, neighbor_state))

        logging.warning("Path Timing Solver: Could not find a valid timing schedule.")
        return None

    def _is_valid_state(self, state: State, path: List[WorldPosition], constraints: set) -> bool:
        """Checks for vertex conflicts at the target state."""
        idx, time = state
        world_pos = path[idx]
        grid_pos = self.coord_manager.world_to_local_grid(world_pos)
        
        if not grid_pos: # Waypoint is outside the manageable grid, treat as valid
            return True
            
        return (grid_pos, time) not in constraints

    def _heuristic(self, current_idx: int, goal_idx: int, path: List[WorldPosition]) -> float:
        """Estimates the minimum time to reach the goal from the current waypoint."""
        remaining_dist_m = 0
        for i in range(current_idx, goal_idx):
            p1_m = self.coord_manager.world_to_local_meters(path[i])
            p2_m = self.coord_manager.world_to_local_meters(path[i+1])
            remaining_dist_m += calculate_distance_3d(p1_m, p2_m)
        
        return remaining_dist_m / DRONE_SPEED_MPS if DRONE_SPEED_MPS > 1e-6 else float('inf')

    def _reconstruct_timed_path(self, came_from: Dict, current_state: State, path: List[WorldPosition]) -> List[Tuple[WorldPosition, int]]:
        timed_path = []
        state = current_state
        while state in came_from:
            idx, time = state
            timed_path.append((path[idx], time))
            state = came_from[state]
        
        # Add the start state
        start_idx, start_time = state
        timed_path.append((path[start_idx], start_time))
        
        return timed_path[::-1]