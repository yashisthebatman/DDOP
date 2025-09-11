# FILE: utils/rrt_star_anytime.py
import random
import logging
import time
from typing import List, Tuple, Optional
import numpy as np

from environment import Environment
from utils.coordinate_manager import CoordinateManager
from utils.geometry import calculate_distance_3d

from config import (
    RRT_GOAL_BIAS, MIN_ALTITUDE, MAX_ALTITUDE,
    RRT_STEP_SIZE_METERS, RRT_NEIGHBORHOOD_RADIUS_METERS
)

class Node:
    """A node in the RRT* tree, storing positions in LOCAL METERS."""
    def __init__(self, position_m: Tuple, parent: 'Node' = None, cost: float = 0.0):
        self.position_m = position_m
        self.parent = parent
        self.cost = cost

class AnytimeRRTStar:
    """
    An anytime implementation of RRT* that finds an initial path quickly and
    continues to improve it until a time budget is exhausted.
    It operates entirely in the local meter coordinate system for geometric consistency.
    """
    def __init__(self, start_world: Tuple, goal_world: Tuple, env: 'Environment', coord_manager: CoordinateManager):
        self.env = env
        self.coord_manager = coord_manager
        
        # Convert start/goal to the canonical meter system immediately
        self.start_pos_m = self.coord_manager.world_to_meters(start_world)
        self.goal_pos_m = self.coord_manager.world_to_meters(goal_world)
        
        self.start_node = Node(self.start_pos_m)
        self.goal_node = Node(self.goal_pos_m)
        self.nodes = [self.start_node]
        self._create_sampling_bounds(start_world, goal_world)

    def _create_sampling_bounds(self, start_world, goal_world):
        """Creates a focused sampling area in world coords for efficiency."""
        # This logic remains in world coords as it's for generating random samples
        center_lon = (start_world[0] + goal_world[0]) / 2
        center_lat = (start_world[1] + goal_world[1]) / 2
        path_dist_lon = abs(start_world[0] - goal_world[0])
        path_dist_lat = abs(start_world[1] - goal_world[1])
        buffer_lon = max(0.01, path_dist_lon * 0.5)
        buffer_lat = max(0.01, path_dist_lat * 0.5)
        self.sample_lon_min = center_lon - (path_dist_lon/2 + buffer_lon)
        self.sample_lon_max = center_lon + (path_dist_lon/2 + buffer_lon)
        self.sample_lat_min = center_lat - (path_dist_lat/2 + buffer_lat)
        self.sample_lat_max = center_lat + (path_dist_lat/2 + buffer_lat)

    def plan(self, time_budget_s: float) -> Tuple[Optional[List[Tuple]], str]:
        """
        Plans a path, returning the best one found within the time budget.
        The returned path is in WORLD coordinates.
        """
        start_time = time.time()
        best_path_m = None
        best_cost = float('inf')
        
        while time.time() - start_time < time_budget_s:
            sample_m = self._get_random_sample_m()
            nearest_node = self._get_nearest_node(sample_m)
            new_node_pos_m = self._steer(nearest_node.position_m, sample_m)
            
            # The collision check must convert back to world coords for the environment API
            from_pos_world = self.coord_manager.meters_to_world(nearest_node.position_m)
            to_pos_world = self.coord_manager.meters_to_world(new_node_pos_m)

            if not self.env.is_line_obstructed(from_pos_world, to_pos_world):
                new_node = Node(new_node_pos_m, parent=nearest_node)
                near_nodes = self._find_near_nodes(new_node)
                best_parent, min_cost = self._choose_best_parent(new_node, near_nodes)
                new_node.parent, new_node.cost = best_parent, min_cost
                self.nodes.append(new_node)
                self._rewire_tree(new_node, near_nodes)

                # Check if this new node can connect to the goal
                new_node_world = self.coord_manager.meters_to_world(new_node.position_m)
                goal_world = self.coord_manager.meters_to_world(self.goal_pos_m)
                if not self.env.is_line_obstructed(new_node_world, goal_world):
                    path_cost = new_node.cost + calculate_distance_3d(new_node.position_m, self.goal_pos_m)
                    if path_cost < best_cost:
                        best_cost = path_cost
                        self.goal_node.parent = new_node
                        best_path_m = self._reconstruct_path(self.goal_node)

        if best_path_m:
            # Convert final path from meters back to world coordinates
            best_path_world = [self.coord_manager.meters_to_world(p) for p in best_path_m]
            return best_path_world, f"Path found successfully (cost: {best_cost:.2f})."
        return None, "No strategic path found within time budget."

    def _get_random_sample_m(self) -> Tuple:
        if random.random() < RRT_GOAL_BIAS: return self.goal_pos_m
        lon = random.uniform(self.sample_lon_min, self.sample_lon_max)
        lat = random.uniform(self.sample_lat_min, self.sample_lat_max)
        alt = random.uniform(MIN_ALTITUDE, MAX_ALTITUDE)
        return self.coord_manager.world_to_meters((lon, lat, alt))

    def _get_nearest_node(self, sample_m: Tuple) -> Node:
        return min(self.nodes, key=lambda n: calculate_distance_3d(n.position_m, sample_m))

    def _steer(self, from_pos_m: Tuple, to_pos_m: Tuple) -> Tuple:
        """Steers from a node towards a sample in METER-SPACE."""
        from_arr, to_arr = np.array(from_pos_m), np.array(to_pos_m)
        direction = to_arr - from_arr
        dist = np.linalg.norm(direction)
        if dist < RRT_STEP_SIZE_METERS:
            return to_pos_m
        new_pos_m = from_arr + (direction / dist) * RRT_STEP_SIZE_METERS
        return tuple(new_pos_m)

    def _find_near_nodes(self, new_node: Node) -> List[Node]:
        return [n for n in self.nodes if calculate_distance_3d(n.position_m, new_node.position_m) <= RRT_NEIGHBORHOOD_RADIUS_METERS]

    def _choose_best_parent(self, new_node: Node, near_nodes: List[Node]) -> Tuple[Node, float]:
        best_parent = new_node.parent
        min_cost = best_parent.cost + calculate_distance_3d(best_parent.position_m, new_node.position_m)
        for p_node in near_nodes:
            dist = calculate_distance_3d(p_node.position_m, new_node.position_m)
            if p_node.cost + dist < min_cost:
                from_world = self.coord_manager.meters_to_world(p_node.position_m)
                to_world = self.coord_manager.meters_to_world(new_node.position_m)
                if not self.env.is_line_obstructed(from_world, to_world):
                    min_cost, best_parent = p_node.cost + dist, p_node
        return best_parent, min_cost

    def _rewire_tree(self, new_node: Node, near_nodes: List[Node]):
        for r_node in near_nodes:
            if r_node == new_node.parent: continue
            dist = calculate_distance_3d(new_node.position_m, r_node.position_m)
            if new_node.cost + dist < r_node.cost:
                from_world = self.coord_manager.meters_to_world(new_node.position_m)
                to_world = self.coord_manager.meters_to_world(r_node.position_m)
                if not self.env.is_line_obstructed(from_world, to_world):
                    r_node.parent, r_node.cost = new_node, new_node.cost + dist

    def _reconstruct_path(self, goal_node: Node) -> List[Tuple]:
        """Reconstructs the path in METER-SPACE."""
        path = []
        current = goal_node
        while current.parent is not None:
            path.append(current.position_m)
            current = current.parent
        path.append(self.start_pos_m)
        return path[::-1]