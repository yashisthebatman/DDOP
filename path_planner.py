import numpy as np
import logging
from typing import Tuple, List, Optional
from collections import defaultdict
import heapq
import time

import config
from environment import Environment
from ml_predictor.predictor import EnergyTimePredictor
from utils.heuristics import TimeHeuristic, EnergyHeuristic, BalancedHeuristic
from utils.jump_point_search import JumpPointSearch # Tactical Engine 1: Speed
from utils.d_star_lite import DStarLite # Tactical Engine 2: Replanning

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
WorldCoord = Tuple[float, float, float]
GridCoord = Tuple[int, int, int]

class PathPlanner3D:
    def __init__(self, env: Environment, predictor: EnergyTimePredictor):
        self.env = env
        self.predictor = predictor
        self.resolution = 25
        self.grid_shape, self.origin_lon, self.origin_lat = self._get_grid_params()
        
        # Sparse cost map for environmental effects (wind, etc.), NOT obstacles
        self.cost_map = {} 
        # self._precompute_environmental_costs() # Optional: add wind pre-computation here if desired

        # --- Strategic Layer: Abstract Graph ---
        self.abstract_graph = defaultdict(dict)
        self.abstract_nodes = {}
        self._build_abstract_graph()
        
        # --- Mission State ---
        self.subgoal_path: Optional[List[WorldCoord]] = None
        self.full_path_segments: Optional[List[List[GridCoord]]] = None
        
        self._calculate_heuristic_baselines()
        logging.info("Hybrid Planner ready. Abstract graph and tactical engines are online.")

    def _build_abstract_graph(self):
        """Constructs the high-level graph of waypoints (hubs, destinations, NFZ corners)."""
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
                if not self.env.is_line_obstructed(pos1, pos2, samples=30):
                    dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                    self.abstract_graph[name1][name2] = dist
                    self.abstract_graph[name2][name1] = dist
        logging.info(f"Abstract graph built in {time.time() - start_time:.2f}s.")

    def _a_star_on_abstract_graph(self, start_pos, end_pos):
        start_name = "动态起点"; self.abstract_nodes[start_name] = start_pos
        end_name = "动态终点"; self.abstract_nodes[end_name] = end_pos
        for name, pos in self.abstract_nodes.items():
            if name not in [start_name, end_name]:
                if not self.env.is_line_obstructed(start_pos, pos): self.abstract_graph[start_name][name] = np.linalg.norm(np.array(start_pos)-np.array(pos))
                if not self.env.is_line_obstructed(end_pos, pos): self.abstract_graph[name][end_name] = np.linalg.norm(np.array(end_pos)-np.array(pos))

        q, came_from, cost = [(0, start_name, [])], {}, {start_name: 0}
        while q:
            _, current, path = heapq.heappop(q)
            if current == end_name: return path + [current]
            for neighbor, weight in self.abstract_graph[current].items():
                new_cost = cost[current] + weight
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    prio = new_cost + np.linalg.norm(np.array(self.abstract_nodes[neighbor])-np.array(self.abstract_nodes[end_name]))
                    heapq.heappush(q, (prio, neighbor, path + [current]))
        return None

    def find_path(self, start_pos: WorldCoord, end_pos: WorldCoord, payload_kg: float, optimization_mode: str, time_weight: float = 0.5) -> Tuple[Optional[List[WorldCoord]], str]:
        """Finds a path using the two-tiered hierarchical planner with JPS for tactical planning."""
        self._build_abstract_graph() # Rebuild to include dynamic obstacles in strategic view
        subgoal_names = self._a_star_on_abstract_graph(start_pos, end_pos)
        if not subgoal_names: return None, "High-level planner failed to find a route."

        self.subgoal_path = [self.abstract_nodes[name] for name in subgoal_names]
        self.full_path_segments = []

        for i in range(len(self.subgoal_path) - 1):
            p1, p2 = self.subgoal_path[i], self.subgoal_path[i+1]
            start_grid, end_grid = self._world_to_grid(p1), self._world_to_grid(p2)
            if start_grid == end_grid: continue

            heuristic = self._get_heuristic(optimization_mode, payload_kg, end_grid, time_weight)
            jps = JumpPointSearch(start_grid, end_grid, self.is_grid_obstructed, heuristic)
            path_segment_grid = jps.search()
            
            if not path_segment_grid: return None, f"Tactical planner (JPS) failed on segment {i}."
            self.full_path_segments.append(path_segment_grid)
        
        return self._stitch_path_segments(), "Hierarchical path found successfully"

    def replan_path(self, drone_pos: WorldCoord, current_segment_idx: int, changed_nfz: list, payload_kg, opt_mode, time_w) -> Tuple[Optional[List[WorldCoord]], str]:
        """Replans a single mission segment using D* Lite."""
        if not self.subgoal_path or not self.full_path_segments: return None, "No active mission to replan."
        logging.info(f"D* Lite activated for segment {current_segment_idx}...")

        # 1. Define the segment to be replanned
        start_grid = self._world_to_grid(drone_pos)
        end_wp = self.subgoal_path[current_segment_idx + 1]
        end_grid = self._world_to_grid(end_wp)

        # 2. Setup D* Lite
        heuristic = self._get_heuristic(opt_mode, payload_kg, end_grid, time_w)
        d_star = DStarLite(start_grid, end_grid, self.cost_map, heuristic)
        
        # 3. Inform D* Lite of cost changes
        changed_cells = self._get_grid_cells_in_nfz(changed_nfz)
        cost_updates = [(cell, float('inf')) for cell in changed_cells]
        for cell, cost in cost_updates: self.cost_map[cell] = cost

        # 4. Compute initial path, then update and replan
        d_star.compute_shortest_path() # Initial computation
        new_segment_grid = d_star.update_and_replan(start_grid, cost_updates)

        if not new_segment_grid: return None, "D* Lite failed to find a repair path."
        
        # 5. Update the master plan with the new segment
        self.full_path_segments[current_segment_idx] = new_segment_grid
        
        return self._stitch_path_segments(start_from_segment=current_segment_idx), "D* Lite replan successful."

    # --- Helper & Utility Methods ---
    def _stitch_path_segments(self, start_from_segment=0) -> List[WorldCoord]:
        """Combines the low-level path segments into one continuous world path."""
        final_path = []
        if start_from_segment == 0:
            final_path.append(self.subgoal_path[0])

        for i in range(start_from_segment, len(self.full_path_segments)):
            segment_world = [self._grid_to_world(p) for p in self.full_path_segments[i]]
            final_path.extend(segment_world[1:])
        return final_path
        
    def _get_heuristic(self, mode, payload, goal_grid, time_w):
        if mode == "time": heuristic_class = TimeHeuristic
        elif mode == "energy": heuristic_class = EnergyHeuristic
        else: heuristic_class = BalancedHeuristic
        return heuristic_class(self, payload, goal_grid, time_w)

    def is_grid_obstructed(self, grid_coord: GridCoord) -> bool:
        if not (0 <= grid_coord[0] < self.grid_shape[0] and 0 <= grid_coord[1] < self.grid_shape[1] and 0 <= grid_coord[2] < self.grid_shape[2]):
            return True
        if self.cost_map.get(grid_coord, 0) == float('inf'): # Quick check for dynamic obstacles
            return True
        world_coord = self._grid_to_world(grid_coord)
        return self.env.is_point_obstructed(world_coord)
    
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