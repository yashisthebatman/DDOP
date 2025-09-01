import numpy as np
import logging
from typing import Tuple, List, Optional
from collections import defaultdict, deque
import heapq
import time
from itertools import product

import config
from environment import Environment
from ml_predictor.predictor import EnergyTimePredictor
from utils.heuristics import TimeHeuristic, EnergyHeuristic, BalancedHeuristic
from utils.jump_point_search import JumpPointSearch
from utils.d_star_lite import DStarLite

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
WorldCoord = Tuple[float, float, float]
GridCoord = Tuple[int, int, int]

class PathPlanner3D:
    def __init__(self, env: Environment, predictor: EnergyTimePredictor):
        self.env = env
        self.predictor = predictor
        self.resolution = 25
        self.grid_shape, self.origin_lon, self.origin_lat = self._get_grid_params()
        
        self.cost_map = {} 
        self.abstract_graph = defaultdict(dict)
        self.abstract_nodes = {}
        self._build_abstract_graph()
        
        self.subgoal_path: Optional[List[WorldCoord]] = None
        self.full_path_segments: Optional[List[List[GridCoord]]] = None
        
        self._calculate_heuristic_baselines()
        
        dummy_goal = (0, 0, 0)
        self.time_h = TimeHeuristic(self, 0.0, dummy_goal)
        self.energy_h = EnergyHeuristic(self, 0.0, dummy_goal)
        self.balanced_h = BalancedHeuristic(self, 0.0, dummy_goal)
        
        self.NEIGHBOR_MOVES = list(product([-1, 0, 1], repeat=3))
        self.NEIGHBOR_MOVES.remove((0, 0, 0))
        
        logging.info("Hybrid Planner ready. Abstract graph and tactical engines are online.")

    # ... _build_abstract_graph and _a_star_on_abstract_graph are unchanged ...
    def _build_abstract_graph(self):
        logging.info("Building abstract waypoint graph...")
        start_time = time.time()
        self.abstract_nodes.clear(); self.abstract_graph.clear()
        
        for name, pos in config.HUBS.items(): self.abstract_nodes[name] = pos
        for name, pos in config.DESTINATIONS.items(): self.abstract_nodes[name] = pos
        for i, zone in enumerate(self.env.get_all_nfzs()):
            corners = [
                (zone[0], zone[1], config.DEFAULT_CRUISING_ALTITUDE), (zone[2], zone[1], config.DEFAULT_CRUISING_ALTITUDE),
                (zone[0], zone[3], config.DEFAULT_CRUISING_ALTITUDE), (zone[2], zone[3], config.DEFAULT_CRUISING_ALTITUDE),
            ]
            for j, corner in enumerate(corners): self.abstract_nodes[f"NFZ_{i}_C_{j}"] = corner

        node_items = list(self.abstract_nodes.items())
        for i in range(len(node_items)):
            for j in range(i + 1, len(node_items)):
                name1, pos1 = node_items[i]; name2, pos2 = node_items[j]
                if not self.env.is_line_obstructed(pos1, pos2):
                    dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                    self.abstract_graph[name1][name2] = dist
                    self.abstract_graph[name2][name1] = dist
        logging.info(f"Abstract graph built in {time.time() - start_time:.2f}s.")

    def _a_star_on_abstract_graph(self, start_pos, end_pos):
        start_name = "dynamic_start"; self.abstract_nodes[start_name] = start_pos
        end_name = "dynamic_end"; self.abstract_nodes[end_name] = end_pos
        
        temp_edges = defaultdict(dict)
        all_nodes = list(self.abstract_nodes.items())
        for name, pos in all_nodes:
             if name != start_name and not self.env.is_line_obstructed(start_pos, pos):
                 temp_edges[start_name][name] = np.linalg.norm(np.array(start_pos)-np.array(pos))
             if name != end_name and not self.env.is_line_obstructed(end_pos, pos):
                 temp_edges[name][end_name] = np.linalg.norm(np.array(end_pos)-np.array(pos))
        if not self.env.is_line_obstructed(start_pos, end_pos):
            temp_edges[start_name][end_name] = np.linalg.norm(np.array(start_pos) - np.array(end_pos))

        graph_view = {**self.abstract_graph, **temp_edges}
        for k,v in temp_edges.items(): graph_view[k] = {**self.abstract_graph.get(k, {}), **v}

        q, cost = [(0, start_name, [])], {start_name: 0}
        while q:
            _, current, path = heapq.heappop(q)
            if current == end_name: return path + [current]
            
            neighbors = {**self.abstract_graph.get(current, {}), **temp_edges.get(current, {})}
            for neighbor, weight in neighbors.items():
                new_cost = cost[current] + weight
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    prio = new_cost + np.linalg.norm(np.array(self.abstract_nodes[neighbor])-np.array(self.abstract_nodes[end_name]))
                    heapq.heappush(q, (prio, neighbor, path + [current]))
        return None
    
    def find_path(self, start_pos: WorldCoord, end_pos: WorldCoord, payload_kg: float, optimization_mode: str, time_weight: float = 0.5) -> Tuple[Optional[List[WorldCoord]], str]:
        self._build_abstract_graph()
        subgoal_names = self._a_star_on_abstract_graph(start_pos, end_pos)
        if not subgoal_names: return None, "High-level planner failed to find a route."

        self.subgoal_path = [self.abstract_nodes[name] for name in subgoal_names]
        self.full_path_segments = []

        for i in range(len(self.subgoal_path) - 1):
            p1, p2 = self.subgoal_path[i], self.subgoal_path[i+1]
            
            start_grid = self._find_nearest_valid_node(self._world_to_grid(p1))
            end_grid = self._find_nearest_valid_node(self._world_to_grid(p2))

            if not start_grid or not end_grid:
                return None, f"Path Error: Could not find valid grid cell for segment {i} start/end."
            
            if start_grid == end_grid: continue

            heuristic = self._get_heuristic(optimization_mode, payload_kg, end_grid, time_weight)
            
            jps = JumpPointSearch(start_grid, end_grid, self.is_grid_obstructed, heuristic)
            path_segment_grid = jps.search()
            
            if not path_segment_grid: return None, f"Tactical planner (JPS) failed on segment {i}."
            self.full_path_segments.append(path_segment_grid)
        
        return self._stitch_path_segments(), "Hierarchical path found successfully"

    def replan_path(self, drone_pos: WorldCoord, current_segment_idx: int, changed_nfz: list, payload_kg, opt_mode, time_w) -> Tuple[Optional[List[WorldCoord]], str]:
        if not self.subgoal_path or not self.full_path_segments: return None, "No active mission to replan."
        logging.info(f"D* Lite activated for segment {current_segment_idx}...")

        start_grid = self._find_nearest_valid_node(self._world_to_grid(drone_pos))
        end_wp = self.subgoal_path[current_segment_idx + 1]
        end_grid = self._find_nearest_valid_node(self._world_to_grid(end_wp))

        if not start_grid or not end_grid:
            return None, "Replan Error: Could not find valid grid cell for drone or subgoal."

        heuristic = self._get_heuristic(opt_mode, payload_kg, end_grid, time_w)
        
        d_star = DStarLite(start_grid, end_grid, self.cost_map, heuristic)
        
        changed_cells = self._get_grid_cells_in_nfz(changed_nfz)
        cost_updates = [(cell, float('inf')) for cell in changed_cells]
        for cell, cost in cost_updates: self.cost_map[cell] = cost

        d_star.compute_shortest_path()
        new_segment_grid = d_star.update_and_replan(start_grid, cost_updates)

        if not new_segment_grid: return None, "D* Lite failed to find a repair path."
        
        self.full_path_segments[current_segment_idx] = new_segment_grid
        
        return self._stitch_path_segments(), "D* Lite replan successful."

    def is_grid_obstructed(self, grid_coord: GridCoord) -> bool:
        """
        ROBUST: Checks if the entire 3D volume of a grid cell intersects any obstacle.
        """
        # 1. Standard out-of-bounds check
        if not (0 <= grid_coord[0] < self.grid_shape[0] and \
                0 <= grid_coord[1] < self.grid_shape[1] and \
                0 <= grid_coord[2] < self.grid_shape[2]):
            return True
            
        # 2. Check sparse cost map for dynamically added obstacles (from D* Lite)
        if self.cost_map.get(grid_coord, 0) == float('inf'):
            return True

        # 3. Perform a volume-based check using the R-tree
        # Calculate the world coordinates of the cell's two opposite corners
        min_corner_world = self._grid_to_world(grid_coord)
        max_corner_world = self._grid_to_world((grid_coord[0] + 1, grid_coord[1] + 1, grid_coord[2] + 1))
        
        # Create a bounding box tuple for the R-tree query
        cell_bounds = (
            min_corner_world[0], min_corner_world[1], min_corner_world[2],
            max_corner_world[0], max_corner_world[1], max_corner_world[2]
        )
        
        # If the number of intersections is > 0, the cell is obstructed
        return self.env.obstacle_index.count(cell_bounds) > 0

    # ... other methods are unchanged ...
    def _find_nearest_valid_node(self, grid_coord: GridCoord) -> Optional[GridCoord]:
        if not self.is_grid_obstructed(grid_coord):
            return grid_coord
        
        q = deque([grid_coord])
        visited = {grid_coord}
        
        while q:
            current = q.popleft()
            for move in self.NEIGHBOR_MOVES:
                neighbor = (current[0] + move[0], current[1] + move[1], current[2] + move[2])
                if neighbor not in visited:
                    visited.add(neighbor)
                    if not self.is_grid_obstructed(neighbor):
                        return neighbor
                    q.append(neighbor)
        return None

    def _stitch_path_segments(self) -> List[WorldCoord]:
        final_path_stitched = []
        for i, segment_grid in enumerate(self.full_path_segments):
            segment_world = [self._grid_to_world(p) for p in segment_grid]
            if i == 0:
                final_path_stitched.extend(segment_world)
            else:
                final_path_stitched.extend(segment_world[1:])
        return final_path_stitched
        
    def _get_heuristic(self, mode, payload, goal_grid, time_w):
        if mode == "time": h = self.time_h
        elif mode == "energy": h = self.energy_h
        else: h = self.balanced_h
        h.goal = goal_grid
        h.payload_kg = payload
        h.time_weight = time_w
        return h
    
    def _get_grid_cells_in_nfz(self, zone: list) -> list:
        min_c=self._world_to_grid((zone[0],zone[1],0));max_c=self._world_to_grid((zone[2],zone[3],config.MAX_ALTITUDE))
        return [(x,y,z) for x in range(min_c[0],max_c[0]+1) for y in range(min_c[1],max_c[1]+1) for z in range(self.grid_shape[2])]

    def _calculate_heuristic_baselines(self):
        self.baseline_time_per_meter=1.0/config.DRONE_SPEED_MPS;p1=(self.origin_lon,self.origin_lat,100);p2_lon=self.origin_lon+self.resolution/(111000*np.cos(np.radians(self.origin_lat)));p2=(p2_lon,self.origin_lat,100);_,e=self.predictor.predict(p1,p2,0,np.array([0,0,0]));self.baseline_energy_per_meter=(e/self.resolution)if e>0 else 0.005
    def _get_grid_params(self):
        lon0,lat0=config.AREA_BOUNDS[0],config.AREA_BOUNDS[1];w=(config.AREA_BOUNDS[2]-lon0)*111000*np.cos(np.radians(lat0));h=(config.AREA_BOUNDS[3]-lat0)*111000;x,y,z=int(w/self.resolution)+1,int(h/self.resolution)+1,int(config.MAX_ALTITUDE/self.resolution)+1;return (x,y,z),lon0,lat0
    def _world_to_grid(self, pos):
        x_m=(pos[0]-self.origin_lon)*111000*np.cos(np.radians(self.origin_lat));y_m=(pos[1]-self.origin_lat)*111000;c=np.clip(np.array([x_m/self.resolution,y_m/self.resolution,pos[2]/self.resolution]),0,np.array(self.grid_shape)-1);return tuple(map(int, c))
    def _grid_to_world(self, g_pos):
        x,y,z=g_pos[0]*self.resolution,g_pos[1]*self.resolution,g_pos[2]*self.resolution;lon=self.origin_lon+x/(111000*np.cos(np.radians(self.origin_lat)));lat=self.origin_lat+y/111000;return(lon,lat,z)