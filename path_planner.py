# ... (imports unchanged)
import logging
from typing import List, Optional, Tuple
import numpy as np
import networkx as nx
from config import AREA_BOUNDS, MIN_ALTITUDE, MAX_ALTITUDE, DRONE_BATTERY_WH
from environment import Environment
from ml_predictor.predictor import EnergyTimePredictor
from utils.coordinate_manager import CoordinateManager
from utils.d_star_lite import DStarLite
from utils.heuristics import HeuristicProvider
from utils.geometry import calculate_distance_3d

class PathPlanner3D:
    # ... (init, build_abstract_graph, _get_cost_function unchanged from previous correct version)
    def __init__(self, env: Environment, predictor: EnergyTimePredictor):
        self.env = env
        self.predictor = predictor
        self.coord_manager = CoordinateManager()
        self.abstract_graph = nx.Graph()
        self.heuristics = HeuristicProvider(self.coord_manager)
    def build_abstract_graph(self, num_nodes_per_axis=10):
        self.abstract_graph.clear()
        lon_min, lat_min, lon_max, lat_max = AREA_BOUNDS
        lons = np.linspace(lon_min, lon_max, num_nodes_per_axis)
        lats = np.linspace(lat_min, lat_max, num_nodes_per_axis)
        alts = np.linspace(MIN_ALTITUDE + 10, MAX_ALTITUDE - 10, 5)
        nodes = []
        for lon in lons:
            for lat in lats:
                for alt in alts:
                    point = (lon, lat, alt)
                    if not self.env.is_point_obstructed(point):
                        nodes.append(point)
        self.abstract_graph.add_nodes_from(nodes)
        max_connection_dist_meters = 1500
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                dist_meters = calculate_distance_3d(
                    self.coord_manager.world_to_local_meters(node1),
                    self.coord_manager.world_to_local_meters(node2)
                )
                if dist_meters < max_connection_dist_meters:
                    if not self.env.is_line_obstructed(node1, node2):
                        self.abstract_graph.add_edge(node1, node2)
        logging.info(f"Abstract graph built: {self.abstract_graph.number_of_nodes()} nodes, {self.abstract_graph.number_of_edges()} edges.")
    def _get_cost_function(self, payload_kg: float, mode: str, time_weight: float = 0.5):
        def cost_func(u, v, d):
            time, energy = self.predictor.predict(u, v, payload_kg, np.array([0,0,0]))
            if time == float('inf'): return float('inf')
            if mode == "time": return time
            elif mode == "energy": return energy
            else:
                norm_time = time / 900.0
                norm_energy = energy / (DRONE_BATTERY_WH / 2.0)
                return (time_weight * norm_time) + ((1 - time_weight) * norm_energy)
        return cost_func

    def find_path(
        self, start_pos: Tuple, end_pos: Tuple, payload_kg: float, mode: str, time_weight: float = 0.5
    ) -> Tuple[Optional[List[Tuple]], str]:
        """Finds the optimal path, now with clearer error messages for obstructed points."""
        if self.abstract_graph.number_of_nodes() == 0:
            return None, "Abstract graph is empty."
        
        # IMPROVEMENT: Check if start/end points are themselves obstructed.
        if self.env.is_point_obstructed(start_pos):
            return None, "Start point is obstructed."
        if self.env.is_point_obstructed(end_pos):
            return None, "Destination point is obstructed."

        start_node = self._find_nearest_node(start_pos)
        end_node = self._find_nearest_node(end_pos)
        
        if not start_node:
            return None, "Could not find a clear path from the start point to the main flight network."
        if not end_node:
             return None, "Could not find a clear path from the main flight network to the destination."

        try:
            # ... (rest of the try block is unchanged)
            cost_func = self._get_cost_function(payload_kg, mode, time_weight)
            heuristic_func = self.heuristics.get_heuristic(mode, end_node, payload_kg, time_weight)
            path_nodes = nx.astar_path(
                self.abstract_graph, source=start_node, target=end_node,
                weight=cost_func, heuristic=heuristic_func
            )
            full_path = [start_pos] + path_nodes + [end_pos]
            simplified_path = self._simplify_path(full_path)
            return simplified_path[1:], "Path found successfully."
        except nx.NetworkXNoPath:
            return None, "No path found between start and end nodes."
        except Exception as e:
            logging.error(f"Strategic planning failed: {e}")
            return None, f"An unexpected error occurred: {e}"
    
    # ... (The rest of path_planner.py is unchanged and correct)
    def replan_path_with_dstar(self, current_pos: Tuple, goal_pos: Tuple, new_obstacle_bounds: Tuple, payload_kg: float, mode: str, time_weight: float = 0.5) -> Tuple[Optional[List[Tuple]], str]:
        self.coord_manager.set_local_grid_origin(current_pos)
        grid_start = self.coord_manager.world_to_local_grid(current_pos)
        grid_goal = self.coord_manager.world_to_local_grid(goal_pos)
        if not grid_start: return None, "Cannot place current position on tactical grid."
        if not grid_goal: return None, "Goal is outside the tactical grid's range for D* Lite."
        cost_map = {}
        min_w = (new_obstacle_bounds[0], new_obstacle_bounds[1], new_obstacle_bounds[2])
        max_w = (new_obstacle_bounds[3], new_obstacle_bounds[4], new_obstacle_bounds[5])
        min_g = self.coord_manager.world_to_local_grid(min_w)
        max_g = self.coord_manager.world_to_local_grid(max_w)
        if min_g and max_g:
            min_gx, max_gx = min(min_g[0], max_g[0]), max(min_g[0], max_g[0])
            min_gy, max_gy = min(min_g[1], max_g[1]), max(min_g[1], max_g[1])
            min_gz, max_gz = min(min_g[2], max_g[2]), max(min_g[2], max_g[2])
            for x in range(min_gx, max_gx + 1):
                for y in range(min_gy, max_gy + 1):
                    for z in range(min_gz, max_gz + 1):
                        if self.coord_manager.is_valid_local_grid_pos((x,y,z)):
                            cost_map[(x,y,z)] = float('inf')
        try:
            dstar = DStarLite(
                start=grid_start, goal=grid_goal, cost_map=cost_map,
                heuristic_provider=self.heuristics, coord_manager=self.coord_manager, mode=mode
            )
            dstar.compute_shortest_path()
            path_grid = dstar.get_path()
            if path_grid:
                path_world = [self.coord_manager.local_grid_to_world(p) for p in path_grid]
                return path_world[1:], "D* Lite replan successful."
            else:
                return None, "D* Lite could not find a path."
        except Exception as e:
            logging.exception(f"D* Lite replanning failed: {e}")
            return None, f"D* Lite error: {e}"
    def invalidate_abstract_graph_edges(self, obstacle_bounds: Tuple):
        edges_to_remove = []
        for u, v in self.abstract_graph.edges():
            if self.env.is_line_obstructed(u, v):
                edges_to_remove.append((u, v))
        self.abstract_graph.remove_edges_from(edges_to_remove)
        logging.info(f"Invalidated and removed {len(edges_to_remove)} edges from the abstract graph.")
    def _find_nearest_node(self, pos: Tuple) -> Optional[Tuple]:
        if not self.abstract_graph: return None
        nodes = np.array(list(self.abstract_graph.nodes()))
        pos_arr = np.array(pos)
        distances = np.linalg.norm(nodes - pos_arr, axis=1)
        for idx in np.argsort(distances):
            node = tuple(nodes[idx])
            if not self.env.is_line_obstructed(pos, node):
                return node
        return None
    def _simplify_path(self, path: List[Tuple]) -> List[Tuple]:
        if len(path) < 3: return path
        simplified_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            for j in range(len(path) - 1, i + 1, -1):
                if not self.env.is_line_obstructed(simplified_path[-1], path[j]):
                    simplified_path.append(path[j])
                    i = j
                    break
            else:
                simplified_path.append(path[i+1])
                i += 1
        return simplified_path