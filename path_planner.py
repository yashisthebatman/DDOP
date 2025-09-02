# ==============================================================================
# File: path_planner.py
# THIS IS THE ONLY FILE WE ARE CHANGING.
# ==============================================================================
import logging
import time
import networkx as nx
import numpy as np
from itertools import product

from config import *
from environment import Environment
from ml_predictor.predictor import EnergyTimePredictor
from utils.geometry import calculate_distance_3d, line_box_intersection
from utils.heuristics import TimeHeuristic, EnergyHeuristic, BalancedHeuristic
from utils.a_star import AStarSearch
from utils.jump_point_search import JumpPointSearch
from utils.d_star_lite import DStarLite
from utils.coordinate_manager import CoordinateManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PathPlanner3D:
    def __init__(self, environment: Environment, predictor: EnergyTimePredictor):
        self.env = environment
        self.predictor = predictor
        self.abstract_graph = None
        self.subgoals = []
        grid_depth = int((MAX_ALTITUDE - MIN_ALTITUDE) / 10)
        self.coord_manager = CoordinateManager(grid_depth=grid_depth)
        self.grid_dims = self.coord_manager.grid_dims
        self.heuristics = {
            "time": TimeHeuristic(self), "energy": EnergyHeuristic(self), "balanced": BalancedHeuristic(self)
        }
        self.cost_map = {}
        # FIX: The constructor is now simple. It only stores dependencies.
        # It does NOT call the complex graph building method anymore.
        # This makes the class robust against simplified mock objects.

    # FIX: A new public method to handle the complex, environment-dependent setup.
    def build_abstract_graph(self):
        """Builds the abstract graph topology. This must be called before find_path."""
        # This check prevents building the graph more than once unnecessarily.
        if self.abstract_graph is not None:
            return

        start_time = time.time()
        logging.info("Building abstract waypoint graph topology...")
        self.abstract_graph = nx.Graph()
        potential_subgoals = list(HUBS.values()) + list(DESTINATIONS.values())
        for nfz in self.env.static_nfzs:
            potential_subgoals.extend([
                (nfz[0], nfz[1], DEFAULT_CRUISING_ALTITUDE), (nfz[2], nfz[3], DEFAULT_CRUISING_ALTITUDE),
                (nfz[0], nfz[3], DEFAULT_CRUISING_ALTITUDE), (nfz[2], nfz[1], DEFAULT_CRUISING_ALTITUDE)
            ])
        self.subgoals = []
        for sg in potential_subgoals:
            if not self.env.is_point_obstructed(sg):
                self.subgoals.append(sg)
            else:
                logging.warning(f"Subgoal {sg} is obstructed and will be excluded from abstract graph.")
        for i, sg in enumerate(self.subgoals): self.abstract_graph.add_node(i, pos=sg)
        for i in range(len(self.subgoals)):
            for j in range(i + 1, len(self.subgoals)):
                p1, p2 = self.subgoals[i], self.subgoals[j]
                if not self.env.is_line_obstructed(p1, p2):
                    dist = calculate_distance_3d(p1, p2)
                    self.abstract_graph.add_edge(i, j, distance=dist)
        logging.info(f"Abstract graph topology built in {time.time() - start_time:.2f}s with {len(self.subgoals)} valid nodes.")

    def find_path(self, start_pos, end_pos, payload, mode, balance_weight=0.5):
        # FIX: Lazily build the graph on the first call to find_path.
        if self.abstract_graph is None:
            self.build_abstract_graph()

        if self.env.is_point_obstructed(start_pos):
            return None, "Error: Start point is located inside an obstacle."
        if self.env.is_point_obstructed(end_pos):
            return None, "Error: Destination point is located inside an obstacle."
        
        # ... The rest of the file from here is unchanged ...
        logging.info(f"Planning with dynamic weights: payload={payload}kg, mode='{mode}'")
        start_node_idx = min(range(len(self.subgoals)), key=lambda i: calculate_distance_3d(self.subgoals[i], start_pos))
        end_node_idx = min(range(len(self.subgoals)), key=lambda i: calculate_distance_3d(self.subgoals[i], end_pos))
        try:
            weight_func = lambda u, v, d: self._calculate_dynamic_edge_weight(u, v, d, payload, mode, balance_weight)
            path_indices = nx.shortest_path(self.abstract_graph, source=start_node_idx, target=end_node_idx, weight=weight_func)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, "No valid path found in the high-level strategic graph."
        waypoints = [start_pos] + [self.subgoals[i] for i in path_indices] + [end_pos]
        waypoints = [waypoints[i] for i in range(len(waypoints)) if i == 0 or np.linalg.norm(np.array(waypoints[i]) - np.array(waypoints[i-1])) > 1e-6]
        full_path = []
        heuristic = self.heuristics.get(mode, self.heuristics['balanced'])
        for i in range(len(waypoints) - 1):
            segment_start, segment_end = waypoints[i], waypoints[i+1]
            start_grid = self.coord_manager.world_to_grid(segment_start)
            end_grid = self.coord_manager.world_to_grid(segment_end)
            safe_start_grid = self._find_nearest_valid_grid_cell(start_grid)
            safe_end_grid = self._find_nearest_valid_grid_cell(end_grid)
            if safe_start_grid is None or safe_end_grid is None:
                return None, f"Could not find a valid grid start/end for segment {i}."
            heuristic.update_params(payload_kg=payload, goal=safe_end_grid, time_weight=balance_weight, start_node=safe_start_grid)
            jps = JumpPointSearch(safe_start_grid, safe_end_grid, self.is_grid_obstructed, heuristic, self.coord_manager)
            path_segment_grid = jps.search()
            if path_segment_grid is None:
                logging.warning(f"JPS failed on segment {i}. Falling back to A*.")
                a_star = AStarSearch(safe_start_grid, safe_end_grid, self.is_grid_obstructed, heuristic, self.coord_manager)
                path_segment_grid = a_star.search()
            if path_segment_grid is None:
                error_msg = f"Fatal Error: Both JPS and A* failed on segment {i}."
                logging.error(error_msg)
                return None, error_msg
            path_segment_world = [self.coord_manager.grid_to_world(p) for p in path_segment_grid]
            if full_path: full_path.extend(path_segment_world[1:])
            else: full_path.extend(path_segment_world)
        simplified_path = self._simplify_path(full_path)
        logging.info(f"Path simplified from {len(full_path)} to {len(simplified_path)} waypoints.")
        return simplified_path, "Path found successfully."

    # (The rest of the file continues unchanged)
    def is_grid_obstructed(self, grid_pos: tuple) -> bool:
        if not self.coord_manager.is_valid_grid_position(grid_pos):
            return True
        world_pos = self.coord_manager.grid_to_world(grid_pos)
        return self.env.is_point_obstructed(world_pos)
    def _calculate_dynamic_edge_weight(self, u, v, d, payload, mode, balance_weight):
        p1 = self.subgoals[u]
        p2 = self.subgoals[v]
        time_cost, energy_cost = self.predictor.fallback_predictor.predict(p1, p2, payload, np.array([0,0,0]))
        if mode == 'time': return time_cost
        elif mode == 'energy': return energy_cost
        else:
            norm_time = time_cost / max(1, d.get('distance', 1) / DRONE_VERTICAL_SPEED_MPS)
            norm_energy = energy_cost / max(1, payload) if payload > 0 else energy_cost
            return norm_time * balance_weight + norm_energy * (1 - balance_weight)
    def _find_nearest_valid_grid_cell(self, grid_pos: tuple) -> tuple:
        if not self.is_grid_obstructed(grid_pos):
            return grid_pos
        logging.warning(f"Initial grid point {grid_pos} is obstructed. Finding nearest valid cell via spiral search...")
        MAX_SEARCH_RADIUS = 50 
        for radius in range(1, MAX_SEARCH_RADIUS + 1):
            for dx, dy, dz in product(range(-radius, radius + 1), repeat=3):
                if abs(dx) != radius and abs(dy) != radius and abs(dz) != radius:
                    continue
                neighbor = (grid_pos[0] + dx, grid_pos[1] + dy, grid_pos[2] + dz)
                if self.coord_manager.is_valid_grid_position(neighbor) and not self.is_grid_obstructed(neighbor):
                    logging.info(f"Found valid alternate cell: {neighbor}")
                    return neighbor
        logging.error(f"Could not find a valid cell within a {MAX_SEARCH_RADIUS}-cell radius of {grid_pos}.")
        return None
    def _simplify_path(self, path_world):
        if len(path_world) < 3:
            return path_world
        simplified_path = [path_world[0]]
        i = 0
        while i < len(path_world) - 1:
            j = i + 2
            while j < len(path_world):
                if self.env.is_line_obstructed(simplified_path[-1], path_world[j]):
                    break
                j += 1
            simplified_path.append(path_world[j-1])
            i = j - 1
        return simplified_path
    def _get_grid_cells_in_bounds(self, world_bounds):
        min_lon, min_lat, min_alt, max_lon, max_lat, max_alt = world_bounds
        min_corner_grid = self.coord_manager.world_to_grid((min_lon, min_lat, min_alt))
        max_corner_grid = self.coord_manager.world_to_grid((max_lon, max_lat, max_alt))
        cells = []
        for x in range(min_corner_grid[0], max_corner_grid[0] + 1):
            for y in range(min_corner_grid[1], max_corner_grid[1] + 1):
                for z in range(min_corner_grid[2], max_corner_grid[2] + 1):
                    cells.append((x, y, z))
        return cells
    def invalidate_abstract_graph_edges(self, nfz_bounds):
        edges_to_remove = []
        for u, v in self.abstract_graph.edges():
            p1 = self.subgoals[u]
            p2 = self.subgoals[v]
            if line_box_intersection(np.array(p1), np.array(p2), nfz_bounds):
                edges_to_remove.append((u, v))
        if edges_to_remove:
            self.abstract_graph.remove_edges_from(edges_to_remove)
            logging.info(f"Strategic replan: Invalidated and removed {len(edges_to_remove)} abstract graph edges due to new obstacle.")
    def replan_path_with_dstar(self, current_pos, end_pos, new_obstacle_bounds, payload, mode, balance_weight=0.5):
        logging.info("Attempting fast replan with D* Lite...")
        try:
            current_grid = self.coord_manager.world_to_grid(current_pos)
            goal_grid = self.coord_manager.world_to_grid(end_pos)
            safe_current_grid = self._find_nearest_valid_grid_cell(current_grid)
            safe_goal_grid = self._find_nearest_valid_grid_cell(goal_grid)
            if not safe_current_grid or not safe_goal_grid:
                return None, "D* Lite failed: could not find valid start/end point."
            heuristic = self.heuristics.get(mode, self.heuristics['balanced'])
            heuristic.update_params(payload, safe_goal_grid, balance_weight, start_node=safe_current_grid)
            d_star = DStarLite(safe_current_grid, safe_goal_grid, self.cost_map, heuristic, self.coord_manager)
            affected_cells = self._get_grid_cells_in_bounds(new_obstacle_bounds)
            logging.info(f"D* Lite processing {len(affected_cells)} cost updates.")
            start_time = time.time()
            new_path_grid = d_star.update_and_replan(safe_current_grid, affected_cells)
            logging.info(f"D* Lite replan completed in {time.time() - start_time:.4f}s")
            if new_path_grid:
                world_path = [self.coord_manager.grid_to_world(p) for p in new_path_grid]
                simplified_path = self._simplify_path(world_path)
                return simplified_path, "D* Lite replan successful."
            else:
                return None, "D* Lite could not find a path."
        except Exception as e:
            logging.error(f"Exception in D* Lite replan: {e}", exc_info=True)
            return None, str(e)