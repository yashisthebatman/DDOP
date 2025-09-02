import numpy as np
import random
import logging
from typing import List, Optional, Tuple

from .coordinate_manager import CoordinateManager
from environment import Environment
from config import AREA_BOUNDS, MIN_ALTITUDE, MAX_ALTITUDE, RRT_ITERATIONS, RRT_STEP_SIZE_METERS, RRT_GOAL_BIAS, RRT_NEIGHBORHOOD_RADIUS_METERS

class Node:
    """A node in the RRT* search tree."""
    def __init__(self, position: Tuple, parent: 'Node' = None, cost: float = 0.0):
        self.position = position
        self.parent = parent
        self.cost = cost

class RRTStar:
    """RRT* algorithm for strategic path planning in a 3D environment."""
    def __init__(self, start: Tuple, goal: Tuple, env: Environment, coord_manager: CoordinateManager):
        self.start_pos = start
        self.goal_pos = goal
        self.env = env
        self.coord_manager = coord_manager
        self.start_node = Node(start)
        self.goal_node = Node(goal)
        self.nodes = [self.start_node]

    def plan(self) -> Tuple[Optional[List[Tuple]], str]:
        """Runs the RRT* planning algorithm."""
        logging.info("Starting RRT* strategic planner...")
        for i in range(RRT_ITERATIONS):
            sample = self._get_random_sample()
            nearest_node = self._get_nearest_node(sample)
            new_node_pos = self._steer(nearest_node.position, sample)
            
            if new_node_pos is None or not self._is_collision_free(nearest_node.position, new_node_pos):
                continue
            
            new_node = Node(new_node_pos, parent=nearest_node)
            near_nodes = self._find_near_nodes(new_node)
            best_parent, min_cost = self._choose_best_parent(new_node, near_nodes)
            new_node.parent = best_parent
            new_node.cost = min_cost
            self.nodes.append(new_node)
            self._rewire_tree(new_node, near_nodes)
        
        self._check_and_connect_goal()
        if not self.goal_node.parent:
            logging.warning("RRT* could not find a path to the goal after all iterations.")
            return None, "No strategic path found."
        
        final_path = self._reconstruct_path(self.goal_node)
        logging.info(f"RRT* found a path with {len(final_path)} waypoints.")
        return final_path, "Path found successfully."

    def _get_random_sample(self) -> Tuple:
        if random.random() < RRT_GOAL_BIAS:
            return self.goal_pos
        lon = random.uniform(AREA_BOUNDS[0], AREA_BOUNDS[2])
        lat = random.uniform(AREA_BOUNDS[1], AREA_BOUNDS[3])
        alt = random.uniform(MIN_ALTITUDE, MAX_ALTITUDE)
        return (lon, lat, alt)

    def _get_nearest_node(self, sample: Tuple) -> Node:
        return min(self.nodes, key=lambda node: np.linalg.norm(np.array(self.coord_manager.world_to_local_meters(node.position)) - np.array(self.coord_manager.world_to_local_meters(sample))))

    def _steer(self, from_pos: Tuple, to_sample: Tuple) -> Optional[Tuple]:
        """Steers from a node towards a sample, respecting step size."""
        from_m = np.array(self.coord_manager.world_to_local_meters(from_pos))
        to_m = np.array(self.coord_manager.world_to_local_meters(to_sample))
        direction = to_m - from_m
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return None
        
        target_dist = min(dist, RRT_STEP_SIZE_METERS)
        offset_m = (direction / dist) * target_dist
        new_pos_m = from_m + offset_m
        
        # Reverse the world_to_local_meters calculation to get the new world coordinate
        lon_min, lat_min, _, _ = self.coord_manager.lon_min, self.coord_manager.lat_min, self.coord_manager.lon_max, self.coord_manager.lat_max
        new_lon = (new_pos_m[0] / self.coord_manager.lon_deg_to_m) + lon_min
        new_lat = (new_pos_m[1] / self.coord_manager.lat_deg_to_m) + lat_min
        new_alt = new_pos_m[2] # Altitude is already in meters
        return (new_lon, new_lat, new_alt)

    def _is_collision_free(self, pos1: Tuple, pos2: Tuple) -> bool:
        return not self.env.is_line_obstructed(pos1, pos2)

    def _find_near_nodes(self, new_node: Node) -> List[Node]:
        new_node_m = np.array(self.coord_manager.world_to_local_meters(new_node.position))
        return [node for node in self.nodes if np.linalg.norm(np.array(self.coord_manager.world_to_local_meters(node.position)) - new_node_m) < RRT_NEIGHBORHOOD_RADIUS_METERS]

    def _choose_best_parent(self, new_node: Node, near_nodes: List[Node]) -> Tuple[Node, float]:
        best_parent = new_node.parent
        new_node_pos_m = np.array(self.coord_manager.world_to_local_meters(new_node.position))
        min_cost = best_parent.cost + np.linalg.norm(new_node_pos_m - np.array(self.coord_manager.world_to_local_meters(best_parent.position)))
        
        for p_node in near_nodes:
            dist = np.linalg.norm(new_node_pos_m - np.array(self.coord_manager.world_to_local_meters(p_node.position)))
            potential_cost = p_node.cost + dist
            if potential_cost < min_cost and self._is_collision_free(p_node.position, new_node.position):
                min_cost = potential_cost
                best_parent = p_node
        return best_parent, min_cost

    def _rewire_tree(self, new_node: Node, near_nodes: List[Node]):
        new_node_m = np.array(self.coord_manager.world_to_local_meters(new_node.position))
        for r_node in near_nodes:
            if r_node == new_node.parent: continue
            dist = np.linalg.norm(new_node_m - np.array(self.coord_manager.world_to_local_meters(r_node.position)))
            potential_new_cost = new_node.cost + dist
            if potential_new_cost < r_node.cost and self._is_collision_free(new_node.position, r_node.position):
                r_node.parent = new_node
                r_node.cost = potential_new_cost
    
    def _check_and_connect_goal(self) -> bool:
        """Checks if the goal is reachable from the nearest node and connects it."""
        nearest_to_goal = self._get_nearest_node(self.goal_pos)
        dist_m = np.linalg.norm(np.array(self.coord_manager.world_to_local_meters(nearest_to_goal.position)) - np.array(self.coord_manager.world_to_local_meters(self.goal_pos)))
        
        if dist_m < RRT_STEP_SIZE_METERS and self._is_collision_free(nearest_to_goal.position, self.goal_pos):
            self.goal_node.parent = nearest_to_goal
            self.goal_node.cost = nearest_to_goal.cost + dist_m
            return True
        return False

    def _reconstruct_path(self, goal_node: Node) -> List[Tuple]:
        path = []
        current = goal_node
        while current is not None:
            path.append(current.position)
            current = current.parent
        path.reverse()
        return path