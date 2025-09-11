import random
import logging
from typing import List, Tuple, Optional
import numpy as np

from environment import Environment
from utils.coordinate_manager import CoordinateManager
from utils.geometry import calculate_distance_3d

from config import (
    RRT_ITERATIONS, RRT_GOAL_BIAS, AREA_BOUNDS, MIN_ALTITUDE, MAX_ALTITUDE,
    RRT_STEP_SIZE_METERS, RRT_NEIGHBORHOOD_RADIUS_METERS
)

class Node:
    def __init__(self, position: Tuple, parent: 'Node' = None, cost: float = 0.0):
        self.position = position
        self.parent = parent
        self.cost = cost

class RRTStar:
    def __init__(self, start: Tuple, goal: Tuple, env: 'Environment', coord_manager: CoordinateManager):
        self.start_pos = start
        self.goal_pos = goal
        self.env = env
        self.coord_manager = coord_manager
        self.start_node = Node(start)
        self.goal_node = Node(goal)
        self.nodes = [self.start_node]
        self._create_sampling_bounds()

    def _create_sampling_bounds(self):
        center_lon = (self.start_pos[0] + self.goal_pos[0]) / 2
        center_lat = (self.start_pos[1] + self.goal_pos[1]) / 2
        path_dist_lon = abs(self.start_pos[0] - self.goal_pos[0])
        path_dist_lat = abs(self.start_pos[1] - self.goal_pos[1])
        buffer_lon = max(0.01, path_dist_lon * 0.5)
        buffer_lat = max(0.01, path_dist_lat * 0.5)
        half_width_lon = path_dist_lon / 2 + buffer_lon
        half_width_lat = path_dist_lat / 2 + buffer_lat
        self.sample_lon_min = max(center_lon - half_width_lon, AREA_BOUNDS[0])
        self.sample_lon_max = min(center_lon + half_width_lon, AREA_BOUNDS[2])
        self.sample_lat_min = max(center_lat - half_width_lat, AREA_BOUNDS[1])
        self.sample_lat_max = min(center_lat + half_width_lat, AREA_BOUNDS[3])

    def plan(self) -> Tuple[Optional[List[Tuple]], str]:
        for i in range(RRT_ITERATIONS):
            sample = self._get_random_sample()
            nearest_node = self._get_nearest_node(sample)
            new_node_pos = self._steer(nearest_node.position, sample)
            if new_node_pos and self._is_collision_free(nearest_node.position, new_node_pos):
                new_node = Node(new_node_pos, parent=nearest_node)
                near_nodes = self._find_near_nodes(new_node)
                best_parent, min_cost = self._choose_best_parent(new_node, near_nodes)
                new_node.parent, new_node.cost = best_parent, min_cost
                self.nodes.append(new_node)
                self._rewire_tree(new_node, near_nodes)
        if not self._connect_goal_to_tree():
            return None, "No strategic path found."
        return self._reconstruct_path(self.goal_node), "Path found successfully."

    def _get_random_sample(self) -> Tuple:
        if random.random() < RRT_GOAL_BIAS: return self.goal_pos
        lon = random.uniform(self.sample_lon_min, self.sample_lon_max)
        lat = random.uniform(self.sample_lat_min, self.sample_lat_max)
        alt = random.uniform(MIN_ALTITUDE, MAX_ALTITUDE)
        return (lon, lat, alt)

    def _get_nearest_node(self, sample: Tuple) -> Node:
        sample_m = np.array(self.coord_manager.world_to_local_meters(sample))
        return min(self.nodes, key=lambda n: np.linalg.norm(np.array(self.coord_manager.world_to_local_meters(n.position)) - sample_m))

    def _steer(self, from_pos: Tuple, to_sample: Tuple) -> Optional[Tuple]:
        from_m = np.array(self.coord_manager.world_to_local_meters(from_pos))
        to_m = np.array(self.coord_manager.world_to_local_meters(to_sample))
        direction = to_m - from_m
        dist = np.linalg.norm(direction)
        if dist < 1e-6: return None
        target_dist = min(dist, RRT_STEP_SIZE_METERS)
        offset_m = (direction / dist) * target_dist
        
        # Convert meter offset back to world coordinates
        new_lon = from_pos[0] + (offset_m[0] / self.coord_manager.lon_deg_to_m)
        new_lat = from_pos[1] + (offset_m[1] / self.coord_manager.lat_deg_to_m)
        new_alt = np.clip(from_pos[2] + offset_m[2], MIN_ALTITUDE, MAX_ALTITUDE)
        return (new_lon, new_lat, new_alt)

    def _is_collision_free(self, p1: Tuple, p2: Tuple) -> bool:
        return not self.env.is_line_obstructed(p1, p2)

    def _find_near_nodes(self, new_node: Node) -> List[Node]:
        new_node_m = np.array(self.coord_manager.world_to_local_meters(new_node.position))
        return [n for n in self.nodes if np.linalg.norm(np.array(self.coord_manager.world_to_local_meters(n.position)) - new_node_m) < RRT_NEIGHBORHOOD_RADIUS_METERS]

    def _choose_best_parent(self, new_node: Node, near_nodes: List[Node]) -> Tuple[Node, float]:
        best_parent = new_node.parent
        new_node_pos_m = np.array(self.coord_manager.world_to_local_meters(new_node.position))
        min_cost = best_parent.cost + np.linalg.norm(new_node_pos_m - np.array(self.coord_manager.world_to_local_meters(best_parent.position)))
        for p_node in near_nodes:
            dist = np.linalg.norm(new_node_pos_m - np.array(self.coord_manager.world_to_local_meters(p_node.position)))
            if p_node.cost + dist < min_cost and self._is_collision_free(p_node.position, new_node.position):
                min_cost, best_parent = p_node.cost + dist, p_node
        return best_parent, min_cost

    def _rewire_tree(self, new_node: Node, near_nodes: List[Node]):
        new_node_m = np.array(self.coord_manager.world_to_local_meters(new_node.position))
        for r_node in near_nodes:
            if r_node == new_node.parent: continue
            dist = np.linalg.norm(new_node_m - np.array(self.coord_manager.world_to_local_meters(r_node.position)))
            if new_node.cost + dist < r_node.cost and self._is_collision_free(new_node.position, r_node.position):
                r_node.parent, r_node.cost = new_node, new_node.cost + dist

    def _connect_goal_to_tree(self) -> bool:
        best_parent = None
        min_cost = float('inf')
        for node in self.nodes:
            if self._is_collision_free(node.position, self.goal_pos):
                dist = calculate_distance_3d(self.coord_manager.world_to_local_meters(node.position), self.coord_manager.world_to_local_meters(self.goal_pos))
                if node.cost + dist < min_cost:
                    min_cost, best_parent = node.cost + dist, node
        if best_parent:
            self.goal_node.parent, self.goal_node.cost = best_parent, min_cost
            return True
        return False

    def _reconstruct_path(self, goal_node: Node) -> List[Tuple]:
        path = [current.position for current in iter(lambda: goal_node, None) if (goal_node := goal_node.parent)]
        path.append(self.start_pos)
        return path[::-1]