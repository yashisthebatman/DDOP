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
from utils.geometry import calculate_distance_3d, point_in_aabb

class PathPlanner3D:
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
        # FIX: Increase the number of altitude layers to create a more connected
        # graph, allowing paths to form over and under complex obstacles.
        alts = np.linspace(MIN_ALTITUDE + 10, MAX_ALTITUDE - 10, 8)

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

    # ... (Rest of path_planner.py is correct and does not need changes)
    def _get_cost_function(self, payload_kg: float, mode: str, time_weight: float = 0.5):
        def cost_func(u, v, d):
            time, energy = self.predictor.predict(u, v, payload_kg, np.array([0,0,0]))
            if time == float('inf'): return float('inf')
            if mode == "time": return time
            elif mode == "energy": return energy
            else:
                norm_time, norm_energy = time / 900.0, energy / (DRONE_BATTERY_WH / 2.0)
                return (time_weight * norm_time) + ((1 - time_weight) * norm_energy)
        return cost_func
    def find_path(self, start_pos: Tuple, end_pos: Tuple, payload_kg: float, mode: str, time_weight: float = 0.5) -> Tuple[Optional[List[Tuple]], str]:
        if self.abstract_graph.number_of_nodes() == 0: return None, "Abstract graph is empty."
        if self.env.is_point_obstructed(start_pos): return None, "Start point is obstructed."
        if self.env.is_point_obstructed(end_pos): return None, "Destination point is obstructed."
        start_node, end_node = self._find_nearest_node(start_pos), self._find_nearest_node(end_pos)
        if not start_node: return None, "Could not find a clear path from the start point to the main flight network."
        if not end_node: return None, "Could not find a clear path from the main flight network to the destination."
        try:
            cost_func = self._get_cost_function(payload_kg, mode, time_weight)
            heuristic_func = self.heuristics.get_heuristic(mode, end_node, payload_kg, time_weight)
            path_nodes = nx.astar_path(self.abstract_graph, source=start_node, target=end_node, weight=cost_func, heuristic=heuristic_func)
            full_path = [start_pos] + path_nodes + [end_pos]
            simplified_path = self._simplify_path(full_path)
            return simplified_path[1:], "Path found successfully."
        except nx.NetworkXNoPath: return None, "No path found between start and end nodes."
        except Exception as e:
            logging.error(f"Strategic planning failed: {e}"); return None, f"An unexpected error occurred: {e}"
    def perform_hybrid_replan(self, current_pos: Tuple, goal_pos: Tuple, new_obstacle_bounds: Tuple, payload_kg: float, mode: str, time_weight: float = 0.5) -> Tuple[Optional[List[Tuple]], str]:
        logging.info("--- Starting Hybrid Replan ---")
        self.invalidate_abstract_graph_edges(new_obstacle_bounds)
        logging.info("Stage 1: Finding a tactical escape route to the safe network.")
        escape_goal_node = self._find_nearest_safe_abstract_node(current_pos, new_obstacle_bounds)
        if not escape_goal_node: return None, "Fatal: Drone is trapped. No safe node on the abstract graph is reachable."
        logging.info(f"Found safe escape goal: {escape_goal_node}")
        escape_path, status = self._find_tactical_path(current_pos, escape_goal_node, new_obstacle_bounds)
        if not escape_path: return None, f"Tactical escape failed: {status}"
        logging.info(f"Tactical escape path with {len(escape_path)} waypoints found.")
        logging.info("Stage 2: Performing strategic replan from the safe escape node.")
        final_dest_node = self._find_nearest_node(goal_pos)
        if not final_dest_node: return None, "Strategic Replan Failed: Could not find network node near final destination."
        try:
            cost_func = self._get_cost_function(payload_kg, mode, time_weight)
            heuristic_func = self.heuristics.get_heuristic(mode, final_dest_node, payload_kg, time_weight)
            strategic_path_nodes = nx.astar_path(self.abstract_graph, source=escape_goal_node, target=final_dest_node, weight=cost_func, heuristic=heuristic_func)
            full_new_path = escape_path + strategic_path_nodes + [goal_pos]
            simplified_path = self._simplify_path(full_new_path)
            logging.info("--- Hybrid Replan Successful ---")
            return simplified_path, "Hybrid replan successful."
        except nx.NetworkXNoPath: return None, "Strategic Replan Failed: No path exists from escape node to destination."
        except Exception as e:
            logging.exception("An unexpected error occurred during strategic replan stage."); return None, f"Strategic replan error: {e}"
    def _find_tactical_path(self, start_pos, end_pos, new_obstacle_bounds):
        self.coord_manager.set_local_grid_origin(start_pos)
        grid_start, grid_goal = self.coord_manager.world_to_local_grid(start_pos), self.coord_manager.world_to_local_grid(end_pos)
        if not grid_start or not grid_goal: return None, "Start or end of tactical path is outside the local grid."
        cost_map = {}
        min_g = self.coord_manager.world_to_local_grid((new_obstacle_bounds[0], new_obstacle_bounds[1], new_obstacle_bounds[2]))
        max_g = self.coord_manager.world_to_local_grid((new_obstacle_bounds[3], new_obstacle_bounds[4], new_obstacle_bounds[5]))
        if min_g and max_g:
            min_gx, max_gx = min(min_g[0], max_g[0]), max(min_g[0], max_g[0])
            min_gy, max_gy = min(min_g[1], max_g[1]), max(min_g[1], max_g[1])
            min_gz, max_gz = min(min_g[2], max_g[2]), max(min_g[2], max_g[2])
            for x in range(min_gx, max_gx + 1):
                for y in range(min_gy, max_gy + 1):
                    for z in range(min_gz, max_gz + 1):
                        if self.coord_manager.is_valid_local_grid_pos((x,y,z)): cost_map[(x,y,z)] = float('inf')
        dstar = DStarLite(start=grid_start, goal=grid_goal, cost_map=cost_map, heuristic_provider=self.heuristics, coord_manager=self.coord_manager, mode='time')
        dstar.compute_shortest_path(); path_grid = dstar.get_path()
        if path_grid:
            path_world = [self.coord_manager.local_grid_to_world(p) for p in path_grid]
            return path_world, "Path found."
        return None, "D* Lite could not find a tactical path."
    def _find_nearest_safe_abstract_node(self, pos, obstacle_bounds):
        if not self.abstract_graph: return None
        nodes = np.array(list(self.abstract_graph.nodes())); pos_arr = np.array(pos)
        distances = np.linalg.norm(nodes - pos_arr, axis=1)
        for idx in np.argsort(distances):
            node = tuple(nodes[idx])
            if not point_in_aabb(node, obstacle_bounds):
                if not self.env.is_line_obstructed(pos, node): return node
        return None
    def invalidate_abstract_graph_edges(self, obstacle_bounds: Tuple):
        edges_to_remove = []
        for u, v in self.abstract_graph.edges():
            if self.env.is_line_obstructed(u, v): edges_to_remove.append((u, v))
        self.abstract_graph.remove_edges_from(edges_to_remove)
        logging.info(f"Invalidated and removed {len(edges_to_remove)} edges from the abstract graph.")
    def _find_nearest_node(self, pos: Tuple) -> Optional[Tuple]:
        if not self.abstract_graph: return None
        nodes = np.array(list(self.abstract_graph.nodes())); pos_arr = np.array(pos)
        distances = np.linalg.norm(nodes - pos_arr, axis=1)
        for idx in np.argsort(distances):
            node = tuple(nodes[idx])
            if not self.env.is_line_obstructed(pos, node): return node
        return None
    def _simplify_path(self, path: List[Tuple]) -> List[Tuple]:
        if len(path) < 3: return path
        simplified_path = [path[0]]; i = 0
        while i < len(path) - 1:
            for j in range(len(path) - 1, i + 1, -1):
                if not self.env.is_line_obstructed(simplified_path[-1], path[j]):
                    simplified_path.append(path[j]); i = j; break
            else: simplified_path.append(path[i+1]); i += 1
        return simplified_path