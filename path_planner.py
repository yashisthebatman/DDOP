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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PathPlanner3D:
    def __init__(self, environment, predictor):
        self.env = environment
        self.predictor = predictor
        self.grid_size = config.GRID_RESOLUTION
        self.abstract_graph = None
        self.subgoals = []
        self.subgoal_path = []
        self.resolution = 25  # Grid resolution in meters
        
        # Grid dimensions
        lon_range = config.AREA_BOUNDS[2] - config.AREA_BOUNDS[0]
        lat_range = config.AREA_BOUNDS[3] - config.AREA_BOUNDS[1]
        self.grid_width = int(lon_range / config.GRID_RESOLUTION) + 1
        self.grid_height = int(lat_range / config.GRID_RESOLUTION) + 1
        self.grid_depth = 20  # Fixed altitude levels
        
        # Heuristics initialization
        self.heuristics = {
            "time": TimeHeuristic(self),
            "energy": EnergyHeuristic(self),
            "balanced": BalancedHeuristic(self)
        }
        
        # Cost map and baseline calculations
        self.cost_map = {}
        self.baseline_energy_per_meter = 0.01  # Default baseline
        self.baseline_time_per_meter = 1.0 / config.DRONE_SPEED_MPS
        
        self._build_abstract_graph()

    def _world_to_grid(self, world_pos):
        """Convert world coordinates to grid coordinates."""
        lon, lat, alt = world_pos
        grid_x = int((lon - config.AREA_BOUNDS[0]) / self.grid_size)
        grid_y = int((lat - config.AREA_BOUNDS[1]) / self.grid_size)
        grid_z = int((alt - config.MIN_ALTITUDE) / ((config.MAX_ALTITUDE - config.MIN_ALTITUDE) / self.grid_depth))
        
        # Clamp to valid ranges
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        grid_z = max(0, min(grid_z, self.grid_depth - 1))
        
        return (grid_x, grid_y, grid_z)

    def _grid_to_world(self, grid_pos):
        """Convert grid coordinates to world coordinates."""
        grid_x, grid_y, grid_z = grid_pos
        
        lon = grid_x * self.grid_size + config.AREA_BOUNDS[0]
        lat = grid_y * self.grid_size + config.AREA_BOUNDS[1]
        alt = config.MIN_ALTITUDE + (grid_z * (config.MAX_ALTITUDE - config.MIN_ALTITUDE) / self.grid_depth)
        
        return (lon, lat, alt)

    def is_grid_obstructed(self, grid_pos):
        """Check if grid position is obstructed."""
        world_pos = self._grid_to_world(grid_pos)
        return self.env.is_point_obstructed(world_pos)

    def _build_abstract_graph(self):
        """Build abstract waypoint graph for strategic planning."""
        start_time = time.time()
        logging.info("Building abstract waypoint graph...")
        self.abstract_graph = nx.Graph()
        
        # Collect all strategic waypoints
        self.subgoals = list(config.HUBS.values()) + list(config.DESTINATIONS.values())
        
        # Add waypoints around no-fly zones
        for nfz in self.env.static_nfzs:
            self.subgoals.extend([
                (nfz[0], nfz[1], config.DEFAULT_CRUISING_ALTITUDE),
                (nfz[2], nfz[3], config.DEFAULT_CRUISING_ALTITUDE),
                (nfz[0], nfz[3], config.DEFAULT_CRUISING_ALTITUDE),
                (nfz[2], nfz[1], config.DEFAULT_CRUISING_ALTITUDE)
            ])

        # Add nodes to graph
        for i, sg in enumerate(self.subgoals):
            self.abstract_graph.add_node(i, pos=sg)

        # Connect unobstructed waypoint pairs
        for i in range(len(self.subgoals)):
            for j in range(i + 1, len(self.subgoals)):
                p1, p2 = self.subgoals[i], self.subgoals[j]
                if not self._is_path_obstructed(p1, p2):
                    time_cost, energy_cost = self.predictor.predict_energy_time(
                        np.array(p1), np.array(p2), 1.0)
                    self.abstract_graph.add_edge(i, j, weight=(time_cost, energy_cost))
        
        logging.info(f"Abstract graph built in {time.time() - start_time:.2f}s.")

    def _is_path_obstructed(self, world_p1, world_p2):
        """Check if path between two world coordinates is obstructed."""
        p1, p2 = np.array(world_p1), np.array(world_p2)
        min_coord, max_coord = np.minimum(p1, p2), np.maximum(p1, p2)
        path_bbox = (*min_coord, *max_coord)
        
        potential_obstacles_ids = list(self.env.obstacle_index.intersection(path_bbox))
        if not potential_obstacles_ids:
            return False

        for obs_id in potential_obstacles_ids:
            obstacle = self.env.get_obstacle_by_id(obs_id)
            if obstacle and line_box_intersection(p1, p2, obstacle.bounds):
                return True
        return False

    def find_path(self, start_pos, end_pos, payload, mode, balance_weight=0.5):
        """Find complete path using hierarchical planning."""
        # Strategic planning: find waypoint sequence
        start_node_idx = min(range(len(self.subgoals)), 
                           key=lambda i: np.linalg.norm(np.array(self.subgoals[i]) - np.array(start_pos)))
        end_node_idx = min(range(len(self.subgoals)), 
                         key=lambda i: np.linalg.norm(np.array(self.subgoals[i]) - np.array(end_pos)))

        try:
            if mode == 'time':
                path_indices = nx.shortest_path(self.abstract_graph, source=start_node_idx, 
                                              target=end_node_idx, weight=lambda u, v, d: d['weight'][0])
            elif mode == 'energy':
                path_indices = nx.shortest_path(self.abstract_graph, source=start_node_idx, 
                                              target=end_node_idx, weight=lambda u, v, d: d['weight'][1])
            else:  # balanced
                path_indices = nx.shortest_path(self.abstract_graph, source=start_node_idx, 
                                              target=end_node_idx, 
                                              weight=lambda u, v, d: d['weight'][0] * balance_weight + d['weight'][1] * (1-balance_weight))
        except nx.NetworkXNoPath:
            return None, "No valid path found in the high-level strategic graph."

        # Create waypoint sequence
        waypoints = [start_pos] + [self.subgoals[i] for i in path_indices 
                                 if i != start_node_idx and i != end_node_idx] + [end_pos]
        
        self.subgoal_path = waypoints
        full_path = []
        heuristic = self.heuristics[mode]

        # Tactical planning: detailed path for each segment
        for i in range(len(waypoints) - 1):
            segment_start, segment_end = waypoints[i], waypoints[i+1]
            start_grid, end_grid = self._world_to_grid(segment_start), self._world_to_grid(segment_end)
            
            # Update heuristic parameters
            heuristic.update_params(payload_kg=payload, goal=end_grid, time_weight=balance_weight)
            
            # Use JPS for tactical planning
            jps = JumpPointSearch(start_grid, end_grid, self.is_grid_obstructed, heuristic, 
                                (self.grid_width, self.grid_height, self.grid_depth))
            path_segment_grid = jps.search()

            if path_segment_grid is None:
                logging.error(f"Tactical planner (JPS) failed on segment {i}.")
                return None, f"Tactical planner (JPS) failed on segment {i}."
            
            # Convert to world coordinates
            path_segment_world = [self._grid_to_world(p) for p in path_segment_grid]
            if full_path:
                full_path.extend(path_segment_world[1:])  # Skip duplicate waypoint
            else:
                full_path.extend(path_segment_world)
                
        return full_path, "Path found successfully."

    def replan_path(self, current_pos, current_subgoal_idx, new_nfz, payload, optimization_mode, balance_weight=0.5):
        """Replan path using D* Lite when new obstacles are detected."""
        try:
            # Update environment with new NFZ
            remaining_waypoints = self.subgoal_path[current_subgoal_idx:]
            
            if remaining_waypoints:
                # Use D* Lite for replanning
                current_grid = self._world_to_grid(current_pos)
                goal_grid = self._world_to_grid(remaining_waypoints[-1])
                
                heuristic = self.heuristics.get(optimization_mode, self.heuristics['balanced'])
                heuristic.update_params(payload, goal_grid, balance_weight)
                
                d_star = DStarLite(current_grid, goal_grid, self.cost_map, heuristic)
                
                # Update costs for new obstacle
                cost_updates = []
                for x in range(self.grid_width):
                    for y in range(self.grid_height):
                        for z in range(self.grid_depth):
                            grid_pos = (x, y, z)
                            if self.is_grid_obstructed(grid_pos):
                                cost_updates.append((grid_pos, float('inf')))
                
                new_path_grid = d_star.update_and_replan(current_grid, cost_updates)
                
                if new_path_grid:
                    new_path_world = [self._grid_to_world(p) for p in new_path_grid]
                    return new_path_world, "Replanning successful"
                else:
                    return None, "D* Lite replanning failed"
            else:
                return None, "No remaining waypoints"
        except Exception as e:
            return None, str(e)