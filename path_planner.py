# ==============================================================================
# File: path_planner.py
# ==============================================================================
import logging
import time
import networkx as nx
import numpy as np
from collections import deque
from itertools import product

from config import *
from environment import Environment
from ml_predictor.predictor import EnergyTimePredictor
from utils.geometry import calculate_distance_3d
from utils.heuristics import TimeHeuristic, EnergyHeuristic, BalancedHeuristic
from utils.a_star import AStarSearch
from utils.jump_point_search import JumpPointSearch
from utils.d_star_lite import DStarLite

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PathPlanner3D:
    def __init__(self, environment: Environment, predictor: EnergyTimePredictor):
        self.env = environment
        self.predictor = predictor
        self.grid_size = GRID_RESOLUTION
        self.abstract_graph = None
        self.subgoals = []
        
        lon_range = AREA_BOUNDS[2] - AREA_BOUNDS[0]
        lat_range = AREA_BOUNDS[3] - AREA_BOUNDS[1]
        self.grid_width = int(lon_range / self.grid_size) + 1
        self.grid_height = int(lat_range / self.grid_size) + 1
        self.grid_depth = int((MAX_ALTITUDE - MIN_ALTITUDE) / 10)
        self.grid_dims = (self.grid_width, self.grid_height, self.grid_depth)

        self.heuristics = {"time": TimeHeuristic(self), "energy": EnergyHeuristic(self), "balanced": BalancedHeuristic(self)}
        self.cost_map = {}
        
        self._build_abstract_graph_topology()

    def _world_to_grid(self, world_pos):
        lon, lat, alt = world_pos
        grid_x = int((lon - AREA_BOUNDS[0]) / self.grid_size)
        grid_y = int((lat - AREA_BOUNDS[1]) / self.grid_size)
        alt_step = (MAX_ALTITUDE - MIN_ALTITUDE) / self.grid_depth
        grid_z = int((alt - MIN_ALTITUDE) / alt_step)
        return tuple(map(int, (max(0, min(grid_x, self.grid_width - 1)), max(0, min(grid_y, self.grid_height - 1)), max(0, min(grid_z, self.grid_depth - 1)))))

    def _grid_to_world(self, grid_pos):
        grid_x, grid_y, grid_z = grid_pos
        lon = grid_x * self.grid_size + AREA_BOUNDS[0]
        lat = grid_y * self.grid_size + AREA_BOUNDS[1]
        alt_step = (MAX_ALTITUDE - MIN_ALTITUDE) / self.grid_depth
        alt = MIN_ALTITUDE + (grid_z * alt_step) + (alt_step / 2)
        return (lon, lat, alt)

    def is_grid_obstructed(self, grid_pos):
        world_pos = self._grid_to_world(grid_pos)
        return self.env.is_point_obstructed(world_pos)

    def _build_abstract_graph_topology(self):
        start_time = time.time()
        logging.info("Building abstract waypoint graph topology...")
        self.abstract_graph = nx.Graph()
        
        # Generate potential subgoals
        potential_subgoals = list(HUBS.values()) + list(DESTINATIONS.values())
        for nfz in self.env.static_nfzs:
            potential_subgoals.extend([
                (nfz[0], nfz[1], DEFAULT_CRUISING_ALTITUDE), (nfz[2], nfz[3], DEFAULT_CRUISING_ALTITUDE),
                (nfz[0], nfz[3], DEFAULT_CRUISING_ALTITUDE), (nfz[2], nfz[1], DEFAULT_CRUISING_ALTITUDE)
            ])
        
        # FIX: Validate subgoals to ensure they are not inside obstacles
        self.subgoals = []
        for sg in potential_subgoals:
            if not self.env.is_point_obstructed(sg):
                self.subgoals.append(sg)
            else:
                logging.warning(f"Subgoal {sg} is obstructed and will be excluded from the abstract graph.")

        for i, sg in enumerate(self.subgoals): self.abstract_graph.add_node(i, pos=sg)
        
        for i in range(len(self.subgoals)):
            for j in range(i + 1, len(self.subgoals)):
                p1, p2 = self.subgoals[i], self.subgoals[j]
                if not self.env.is_line_obstructed(p1, p2):
                    dist = calculate_distance_3d(p1, p2)
                    self.abstract_graph.add_edge(i, j, distance=dist)
        logging.info(f"Abstract graph topology built in {time.time() - start_time:.2f}s with {len(self.subgoals)} valid nodes.")

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

    def _find_nearest_valid_grid_cell(self, grid_pos):
        if not self.is_grid_obstructed(grid_pos): return grid_pos
        logging.warning(f"Start/end point {grid_pos} is obstructed. Finding nearest valid cell...")
        q = deque([grid_pos]); visited = {grid_pos}
        for _ in range(500): # Safety limit
            if not q: break
            x, y, z = q.popleft()
            for dx, dy, dz in product([-1, 0, 1], repeat=3):
                if dx==0 and dy==0 and dz==0: continue
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor in visited: continue
                if not (0 <= neighbor[0] < self.grid_width and 0 <= neighbor[1] < self.grid_height and 0 <= neighbor[2] < self.grid_depth): continue
                if not self.is_grid_obstructed(neighbor):
                    logging.info(f"Found valid alternate cell: {neighbor}")
                    return neighbor
                q.append(neighbor); visited.add(neighbor)
        return None
    
    def _simplify_path(self, path_world):
        """Removes unnecessary waypoints if a direct line is clear."""
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


    def find_path(self, start_pos, end_pos, payload, mode, balance_weight=0.5):
        # FIX: Pre-flight check for start/end points inside obstacles
        if self.env.is_point_obstructed(start_pos):
            return None, "Error: Start point is located inside an obstacle."
        if self.env.is_point_obstructed(end_pos):
            return None, "Error: Destination point is located inside an obstacle."

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
            safe_start_grid = self._world_to_grid(segment_start)
            safe_end_grid = self._world_to_grid(segment_end)

            heuristic.update_params(payload_kg=payload, goal=safe_end_grid, time_weight=balance_weight, start_node=safe_start_grid)
            
            jps = JumpPointSearch(safe_start_grid, safe_end_grid, self.is_grid_obstructed, heuristic, self.grid_dims)
            path_segment_grid = jps.search()
            
            if path_segment_grid is None:
                logging.warning(f"JPS failed on segment {i}. Falling back to A*.")
                a_star = AStarSearch(safe_start_grid, safe_end_grid, self.is_grid_obstructed, heuristic, self.grid_dims)
                path_segment_grid = a_star.search()
            
            if path_segment_grid is None:
                error_msg = f"Fatal Error: Both JPS and A* failed on segment {i}."
                logging.error(error_msg)
                return None, error_msg
            
            path_segment_world = [self._grid_to_world(p) for p in path_segment_grid]
            if full_path: full_path.extend(path_segment_world[1:])
            else: full_path.extend(path_segment_world)
                
        simplified_path = self._simplify_path(full_path)
        logging.info(f"Path simplified from {len(full_path)} to {len(simplified_path)} waypoints.")
        return simplified_path, "Path found successfully."

    # FIX: Optimized method to get only affected grid cells for D*
    def _get_grid_cells_in_bounds(self, world_bounds):
        """Gets all grid cells within a given world coordinate bounding box."""
        min_lon, min_lat, min_alt, max_lon, max_lat, max_alt = world_bounds
        min_corner_grid = self._world_to_grid((min_lon, min_lat, min_alt))
        max_corner_grid = self._world_to_grid((max_lon, max_lat, max_alt))

        cells = []
        for x in range(min_corner_grid[0], max_corner_grid[0] + 1):
            for y in range(min_corner_grid[1], max_corner_grid[1] + 1):
                for z in range(min_corner_grid[2], max_corner_grid[2] + 1):
                    cells.append((x, y, z))
        return cells


    def invalidate_abstract_graph_edges(self, nfz_bounds):
        """Checks abstract graph edges against a new obstacle and removes them if obstructed."""
        edges_to_remove = []
        for u, v in self.abstract_graph.edges():
            p1 = self.subgoals[u]
            p2 = self.subgoals[v]
            # Check against the new obstacle bounds directly
            if line_box_intersection(np.array(p1), np.array(p2), nfz_bounds):
                edges_to_remove.append((u, v))
        
        if edges_to_remove:
            self.abstract_graph.remove_edges_from(edges_to_remove)
            logging.info(f"Strategic replan: Invalidated and removed {len(edges_to_remove)} abstract graph edges due to new obstacle.")

    def replan_path_with_dstar(self, current_pos, end_pos, new_obstacle_bounds, payload, mode, balance_weight=0.5):
        logging.info("Attempting fast replan with D* Lite...")
        try:
            safe_current_grid = self._find_nearest_valid_grid_cell(self._world_to_grid(current_pos))
            safe_goal_grid = self._find_nearest_valid_grid_cell(self._world_to_grid(end_pos))
            if not safe_current_grid or not safe_goal_grid: return None, "D* Lite failed: could not find valid start/end point."

            heuristic = self.heuristics.get(mode, self.heuristics['balanced'])
            heuristic.update_params(payload, safe_goal_grid, balance_weight, start_node=safe_current_grid)
            
            d_star = DStarLite(safe_current_grid, safe_goal_grid, self.cost_map, heuristic, self.grid_dims)
            
            # FIX: Use the optimized method to get only affected cells
            affected_cells = self._get_grid_cells_in_bounds(new_obstacle_bounds)
            logging.info(f"D* Lite processing {len(affected_cells)} cost updates.")
            
            start_time = time.time()
            new_path_grid = d_star.update_and_replan(safe_current_grid, affected_cells)
            logging.info(f"D* Lite replan completed in {time.time() - start_time:.4f}s")
            
            if new_path_grid:
                world_path = [self._grid_to_world(p) for p in new_path_grid]
                simplified_path = self._simplify_path(world_path)
                return simplified_path, "D* Lite replan successful."
            else:
                return None, "D* Lite could not find a path."
        except Exception as e:
            logging.error(f"Exception in D* Lite replan: {e}", exc_info=True)
            return None, str(e)