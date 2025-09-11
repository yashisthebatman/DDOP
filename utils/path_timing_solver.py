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
MAX_TIME_STEPS = 500 # Safeguard against infinite searches

class PathTimingSolver:
    """
    Finds a valid timing schedule for a given geometric path, respecting
    space-time constraints from a CBS planner.
    """
    def __init__(self, coord_manager: CoordinateManager):
        self.coord_manager = coord_manager

    def _get_consecutive_waits(self, state: State, came_from: Dict[State, State]) -> int:
        wait_count = 0
        current_s = state
        while current_s in came_from:
            parent_s = came_from[current_s]
            if parent_s[0] == current_s[0]:
                wait_count += 1
                current_s = parent_s
            else:
                break
        return wait_count

    def find_timing(self, geometric_path: List[WorldPosition], constraints: List[Constraint]) -> Optional[List[Tuple[WorldPosition, int]]]:
        if not geometric_path:
            return []

        start_state: State = (0, 0)
        goal_waypoint_idx = len(geometric_path) - 1
        constraint_set = {(c.position, c.timestamp) for c in constraints}

        open_set = [(0, start_state)]
        came_from: Dict[State, State] = {start_state: None} # Mark start
        g_score: Dict[State, int] = {start_state: 0}

        while open_set:
            _, current_state = heapq.heappop(open_set)
            current_idx, current_time = current_state

            if current_idx == goal_waypoint_idx:
                # FIX: Use the new, more reliable forward reconstruction method.
                return self._reconstruct_timed_path_forward(came_from, current_state, geometric_path)

            if current_time > MAX_TIME_STEPS:
                continue

            # Action: Move
            if current_idx < goal_waypoint_idx:
                p1_world, p2_world = geometric_path[current_idx], geometric_path[current_idx + 1]
                p1_m, p2_m = self.coord_manager.world_to_local_meters(p1_world), self.coord_manager.world_to_local_meters(p2_world)
                dist_m = calculate_distance_3d(p1_m, p2_m)
                time_to_travel = int(np.ceil(dist_m / DRONE_SPEED_MPS)) if DRONE_SPEED_MPS > 1e-6 else float('inf')
                
                neighbor_time = current_time + max(1, time_to_travel)
                neighbor_idx = current_idx + 1
                neighbor_state: State = (neighbor_idx, neighbor_time)
                
                if self._is_valid_state(neighbor_state, geometric_path, constraint_set) and neighbor_time < g_score.get(neighbor_state, float('inf')):
                    came_from[neighbor_state] = current_state
                    g_score[neighbor_state] = neighbor_time
                    f_score = neighbor_time + self._heuristic(neighbor_idx, goal_waypoint_idx, geometric_path)
                    heapq.heappush(open_set, (f_score, neighbor_state))

            # Action: Wait
            if self._get_consecutive_waits(current_state, came_from) < MAX_WAIT_TIME:
                neighbor_time = current_time + 1
                neighbor_idx = current_idx
                neighbor_state: State = (neighbor_idx, neighbor_time)
                
                if self._is_valid_state(neighbor_state, geometric_path, constraint_set) and neighbor_time < g_score.get(neighbor_state, float('inf')):
                    came_from[neighbor_state] = current_state
                    g_score[neighbor_state] = neighbor_time
                    f_score = neighbor_time + self._heuristic(neighbor_idx, goal_waypoint_idx, geometric_path)
                    heapq.heappush(open_set, (f_score, neighbor_state))

        logging.warning(f"Path Timing Solver: Could not find a valid timing schedule.")
        return None

    def _is_valid_state(self, state: State, path: List[WorldPosition], constraints: set) -> bool:
        idx, time = state
        world_pos = path[idx]
        grid_pos = self.coord_manager.world_to_local_grid(world_pos)
        return (grid_pos, time) not in constraints if grid_pos else True

    def _heuristic(self, current_idx: int, goal_idx: int, path: List[WorldPosition]) -> float:
        if current_idx >= goal_idx: return 0
        remaining_dist_m = sum(calculate_distance_3d(self.coord_manager.world_to_local_meters(path[i]), self.coord_manager.world_to_local_meters(path[i+1])) for i in range(current_idx, goal_idx))
        return remaining_dist_m / DRONE_SPEED_MPS if DRONE_SPEED_MPS > 1e-6 else float('inf')

    # NEW, ROBUST RECONSTRUCTION METHOD
    def _reconstruct_timed_path_forward(self, came_from: Dict, goal_state: State, path: List[WorldPosition]) -> List[Tuple[WorldPosition, int]]:
        """Reconstructs the path by tracing back to start, then building it forwards."""
        # First, trace back to get the sequence of states in reverse order
        reverse_path_states = []
        curr = goal_state
        while curr is not None:
            reverse_path_states.append(curr)
            curr = came_from[curr]
        
        # Now, build the final timed path by iterating forwards
        timed_path = []
        path_states = reverse_path_states[::-1] # This is the correct chronological order of states
        for state in path_states:
            idx, time = state
            timed_path.append((path[idx], time))
            
        return timed_path