# path_planner.py (Full and Corrected)

import logging
import time
import networkx as nx
import numpy as np
from collections import deque

import config
from utils.d_star_lite import DStarLite
from utils.jump_point_search import JumpPointSearch
from utils.heuristics import TimeHeuristic, EnergyHeuristic, BalancedHeuristic
from utils.geometry import line_box_intersection

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PathPlanner3D:
    def __init__(self, environment, predictor):
        self.env = environment
        self.predictor = predictor
        self.grid_size = config.GRID_RESOLUTION
        self.jps_engine = JumpPointSearch
        self.d_star_engine = DStarLite
        self.abstract_graph = None
        self.subgoals = []

        # --- CORRECTED INITIALIZATION ---
        # Heuristics are created once without mission-specific data.
        self.heuristics = {
            "time": TimeHeuristic(self),
            "energy": EnergyHeuristic(self),
            "balanced": BalancedHeuristic(self)
        }
        
        self._build_abstract_graph()
        # Cost map can be built here or lazily if needed
        self.cost_map = {} 

    def _world_to_grid(self, world_pos):
        lon, lat, alt = world_pos
        grid_x = int((lon - config.AREA_BOUNDS[0]) / self.grid_size)
        grid_y = int((lat - config.AREA_BOUNDS[1]) / self.grid_size)
        grid_z = int(alt / self.grid_size)
        return (grid_x, grid_y, grid_z)

    def _grid_to_world(self, grid_pos):
        grid_x, grid_y, grid_z = grid_pos
        lon = grid_x * self.grid_size + config.AREA_BOUNDS[0]
        lat = grid_y * self.grid_size + config.AREA_BOUNDS[1]
        alt = grid_z * self.grid_size
        return (lon, lat, alt)

    def is_grid_obstructed(self, grid_pos):
        world_pos = self._grid_to_world(grid_pos)
        cell_bounds = (
            world_pos[0], world_pos[1], world_pos[2],
            world_pos[0] + self.grid_size, world_pos[1] + self.grid_size, world_pos[2] + self.grid_size
        )
        return self.env.obstacle_index.count(cell_bounds) > 0

    def _build_abstract_graph(self):
        start_time = time.time()
        logging.info("Building abstract waypoint graph...")
        self.abstract_graph = nx.Graph()
        
        self.subgoals = list(config.HUBS.values()) + list(config.DESTINATIONS.values())
        for nfz in self.env.static_nfzs:
            self.subgoals.append((nfz[0], nfz[1], config.DEFAULT_CRUISING_ALTITUDE))
            self.subgoals.append((nfz[2], nfz[3], config.DEFAULT_CRUISING_ALTITUDE))
            self.subgoals.append((nfz[0], nfz[3], config.DEFAULT_CRUISING_ALTITUDE))
            self.subgoals.append((nfz[2], nfz[1], config.DEFAULT_CRUISING_ALTITUDE))

        for i, sg in enumerate(self.subgoals):
            self.abstract_graph.add_node(i, pos=sg)

        for i in range(len(self.subgoals)):
            for j in range(i + 1, len(self.subgoals)):
                p1 = self.subgoals[i]
                p2 = self.subgoals[j]
                if not self._is_path_obstructed(p1, p2):
                    time_cost, energy_cost = self.predictor.predict_energy_time(np.array(p1), np.array(p2), 1.0)
                    self.abstract_graph.add_edge(i, j, weight=(time_cost, energy_cost))
        
        logging.info(f"Abstract graph built in {time.time() - start_time:.2f}s.")

    def _is_path_obstructed(self, world_p1, world_p2):
        p1 = np.array(world_p1)
        p2 = np.array(world_p2)
        min_coord = np.minimum(p1, p2)
        max_coord = np.maximum(p1, p2)
        path_bbox = (min_coord[0], min_coord[1], min_coord[2], max_coord[0], max_coord[1], max_coord[2])
        
        potential_obstacles_ids = list(self.env.obstacle_index.intersection(path_bbox))
        if not potential_obstacles_ids:
            return False

        for obs_id in potential_obstacles_ids:
            obstacle = self.env.get_obstacle_by_id(obs_id)
            if obstacle and line_box_intersection(p1, p2, obstacle.bounds):
                return True
        return False

    def find_path(self, start_pos, end_pos, payload, mode, balance_weight=0.5):
        start_node_idx = min(range(len(self.subgoals)), key=lambda i: np.linalg.norm(np.array(self.subgoals[i]) - np.array(start_pos)))
        end_node_idx = min(range(len(self.subgoals)), key=lambda i: np.linalg.norm(np.array(self.subgoals[i]) - np.array(end_pos)))

        try:
            if mode == 'time':
                path_indices = nx.shortest_path(self.abstract_graph, source=start_node_idx, target=end_node_idx, weight=lambda u, v, d: d['weight'][0])
            elif mode == 'energy':
                path_indices = nx.shortest_path(self.abstract_graph, source=start_node_idx, target=end_node_idx, weight=lambda u, v, d: d['weight'][1])
            else:
                 path_indices = nx.shortest_path(self.abstract_graph, source=start_node_idx, target=end_node_idx, weight=lambda u, v, d: d['weight'][0] * balance_weight + d['weight'][1] * (1-balance_weight))
        except nx.NetworkXNoPath:
            return None, "No valid path found in the high-level strategic graph."

        waypoints = [start_pos] + [self.subgoals[i] for i in path_indices if i != start_node_idx and i != end_node_idx] + [end_pos]
        
        full_path = []
        heuristic = self.heuristics[mode]

        for i in range(len(waypoints) - 1):
            segment_start = waypoints[i]
            segment_end = waypoints[i+1]
            
            start_grid = self._world_to_grid(segment_start)
            end_grid = self._world_to_grid(segment_end)
            
            # --- CORRECTED: Update heuristic with current mission parameters ---
            heuristic.update_params(payload_kg=payload, goal=end_grid, time_weight=balance_weight)
            
            jps = self.jps_engine(start_grid, end_grid, self.is_grid_obstructed, heuristic)
            path_segment_grid = jps.search()

            if path_segment_grid is None:
                logging.error(f"Tactical planner (JPS) failed on segment {i}.")
                return None, f"Tactical planner (JPS) failed on segment {i}."
            
            path_segment_world = [self._grid_to_world(p) for p in path_segment_grid]
            if full_path:
                full_path.extend(path_segment_world[1:])
            else:
                full_path.extend(path_segment_world)
                
        return full_path, "Path found successfully."

    def replan_path(self, current_pos, current_path, goal_pos, new_obstacles):
        pass # Placeholder