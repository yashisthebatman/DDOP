# ==============================================================================
# optimization/qubo_solver.py
# ==============================================================================
"""
QUBO-based path optimization for drone pathfinding.
This module implements a hybrid A*+QUBO approach where:
1. A* provides initial feasible paths 
2. QUBO optimizes path segments for energy/time efficiency
"""

import numpy as np
import pyqubo
from pyqubo import Binary, Placeholder
from dwave.samplers import SimulatedAnnealingSampler
import networkx as nx
from typing import List, Tuple, Dict, Optional
import logging

from utils.geometry import calculate_distance_3d
from utils.heuristics import a_star_search

class QUBOPathOptimizer:
    """
    QUBO-based path optimizer that refines A* paths for optimal energy/time trade-offs.
    """
    
    def __init__(self, grid, moves, predictor, env, world_converter=None):
        self.grid = grid
        self.moves = moves
        self.predictor = predictor
        self.env = env
        self.sampler = SimulatedAnnealingSampler()
        self.world_converter = world_converter  # Function to convert grid to world coords
        
    def _create_path_segments(self, path_coords: List[Tuple], segment_size: int = 5) -> List[List[Tuple]]:
        """Split path into segments for QUBO optimization."""
        segments = []
        for i in range(0, len(path_coords), segment_size):
            segment = path_coords[i:i + segment_size]
            if len(segment) > 1:  # Need at least 2 points for a segment
                segments.append(segment)
        return segments
    
    def _get_alternative_paths(self, start_coord: Tuple, end_coord: Tuple, 
                              max_alternatives: int = 3) -> List[List[Tuple]]:
        """Generate alternative paths between two points for QUBO selection."""
        paths = []
        
        # Get the primary A* path
        primary_path = a_star_search(start_coord, end_coord, self.grid, self.moves)
        if primary_path:
            paths.append(primary_path)
        
        # Generate alternative paths by modifying the heuristic
        for i in range(max_alternatives - 1):
            # Add some randomization to the heuristic to get different paths
            def modified_heuristic(a, b):
                base = abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
                # Add small random factor to encourage path diversity
                return base + np.random.uniform(-0.5, 0.5)
            
            alt_path = a_star_search(start_coord, end_coord, self.grid, self.moves, 
                                   heuristic_func=modified_heuristic)
            if alt_path and alt_path not in paths:
                paths.append(alt_path)
                
        return paths
    
    def _build_segment_qubo(self, start_coord: Tuple, end_coord: Tuple, 
                           payload_kg: float, weights: Dict) -> pyqubo.Model:
        """Build QUBO model for optimizing a path segment."""
        
        # Get alternative paths for this segment
        alternative_paths = self._get_alternative_paths(start_coord, end_coord)
        
        if len(alternative_paths) <= 1:
            # If only one path available, no optimization needed
            return None, alternative_paths
        
        # Create binary variables for each path
        path_vars = {}
        for i, path in enumerate(alternative_paths):
            path_vars[f'path_{i}'] = Binary(f'path_{i}')
        
        # Constraint: exactly one path must be selected
        path_constraint = sum(path_vars.values()) - 1
        
        # Objective: minimize weighted cost of selected path
        objective = 0
        for i, path in enumerate(alternative_paths):
            # Convert grid path to world coordinates for cost calculation
            if self.world_converter:
                world_path = [self.world_converter(coord) for coord in path]
            else:
                world_path = [self._grid_to_world(coord) for coord in path]
            
            # Calculate time and energy cost
            total_time = 0
            total_energy = 0
            
            for j in range(len(world_path) - 1):
                p1, p2 = world_path[j], world_path[j + 1]
                wind = self.env.weather.get_wind_at_location(p1[0], p1[1])
                
                time_cost, energy_cost = self.predictor.predict(
                    p1, p2, payload_kg, wind,
                    world_path[j-1] if j > 0 else None
                )
                
                if time_cost == float('inf'):
                    # Penalize infeasible paths heavily
                    total_time = 1e6
                    total_energy = 1e6
                    break
                    
                total_time += time_cost
                total_energy += energy_cost
            
            # Weighted cost for this path
            path_cost = weights['time'] * total_time + weights['energy'] * total_energy
            objective += path_vars[f'path_{i}'] * path_cost
        
        # Build the QUBO model
        H = objective + Placeholder('lambda1') * path_constraint**2
        
        model = H.compile()
        return model, alternative_paths
    
    def _grid_to_world(self, grid_pos: Tuple) -> Tuple:
        """Convert grid coordinates to world coordinates."""
        # This should match the conversion in PathPlanner3D
        resolution = 5  # Should be configurable
        origin_lon, origin_lat = -74.02, 40.70  # From AREA_BOUNDS
        
        x_m = grid_pos[0] * resolution
        y_m = grid_pos[1] * resolution
        z_m = grid_pos[2] * resolution
        
        lon = origin_lon + x_m / (111000 * np.cos(np.radians(origin_lat)))
        lat = origin_lat + y_m / 111000
        
        return (lon, lat, z_m)
    
    def optimize_path_segment(self, start_coord: Tuple, end_coord: Tuple,
                            payload_kg: float, weights: Dict) -> Optional[List[Tuple]]:
        """Optimize a single path segment using QUBO."""
        
        model, alternative_paths = self._build_segment_qubo(start_coord, end_coord, 
                                                          payload_kg, weights)
        
        if model is None:
            # Return the only available path
            return alternative_paths[0] if alternative_paths else None
        
        # Sample the QUBO
        feed_dict = {'lambda1': 100.0}  # Constraint weight
        bqm = model.to_bqm(feed_dict=feed_dict)
        
        # Use simulated annealing to solve
        response = self.sampler.sample(bqm, num_reads=100)
        
        # Get the best solution
        best_sample = response.first.sample
        
        # Find which path was selected
        selected_path_idx = None
        for var_name, value in best_sample.items():
            if value == 1 and var_name.startswith('path_'):
                selected_path_idx = int(var_name.split('_')[1])
                break
        
        if selected_path_idx is not None and selected_path_idx < len(alternative_paths):
            return alternative_paths[selected_path_idx]
        
        # Fallback to first path if QUBO solution is invalid
        return alternative_paths[0] if alternative_paths else None
    
    def hybrid_optimize_path(self, start_coord: Tuple, end_coord: Tuple,
                           payload_kg: float, weights: Dict, 
                           segment_size: int = 5) -> Optional[List[Tuple]]:
        """
        Hybrid A*+QUBO path optimization.
        
        1. Use A* to get initial feasible path
        2. Split path into segments
        3. Use QUBO to optimize each segment
        4. Combine optimized segments
        """
        
        # Step 1: Get initial A* path
        initial_path = a_star_search(start_coord, end_coord, self.grid, self.moves)
        
        if not initial_path:
            logging.warning("A* failed to find initial path")
            return None
        
        # For short paths, skip QUBO optimization
        if len(initial_path) <= segment_size:
            return initial_path
        
        # Step 2: Split into segments
        segments = self._create_path_segments(initial_path, segment_size)
        
        # Step 3: Optimize each segment with QUBO
        optimized_path = []
        
        for i, segment in enumerate(segments):
            if len(segment) < 2:
                continue
                
            start_seg = segment[0]
            end_seg = segment[-1]
            
            # Optimize this segment
            optimized_segment = self.optimize_path_segment(
                start_seg, end_seg, payload_kg, weights
            )
            
            if optimized_segment:
                # Add segment to path, avoiding duplicates at connection points
                if not optimized_path:
                    optimized_path.extend(optimized_segment)
                else:
                    # Skip first point to avoid duplication
                    optimized_path.extend(optimized_segment[1:])
            else:
                # Fallback to original segment
                if not optimized_path:
                    optimized_path.extend(segment)
                else:
                    optimized_path.extend(segment[1:])
        
        return optimized_path if optimized_path else initial_path


class HybridPathPlanner:
    """
    Wrapper class that provides a unified interface for hybrid A*+QUBO planning.
    """
    
    def __init__(self, grid, moves, predictor, env, world_converter=None):
        self.qubo_optimizer = QUBOPathOptimizer(grid, moves, predictor, env, world_converter)
        self.grid = grid
        self.moves = moves
        
    def find_optimal_path(self, start_coord: Tuple, end_coord: Tuple,
                         payload_kg: float, weights: Dict, 
                         use_qubo: bool = True) -> Optional[List[Tuple]]:
        """
        Find optimal path using hybrid A*+QUBO approach.
        
        Args:
            start_coord: Starting grid coordinate
            end_coord: Ending grid coordinate  
            payload_kg: Payload weight
            weights: Optimization weights {'time': float, 'energy': float}
            use_qubo: Whether to use QUBO optimization (False for A*-only)
        
        Returns:
            List of grid coordinates representing optimal path
        """
        
        if not use_qubo:
            # Fallback to pure A* search
            return a_star_search(start_coord, end_coord, self.grid, self.moves)
        
        try:
            # Use hybrid A*+QUBO optimization
            return self.qubo_optimizer.hybrid_optimize_path(
                start_coord, end_coord, payload_kg, weights
            )
        except Exception as e:
            logging.warning(f"QUBO optimization failed: {e}. Falling back to A*")
            return a_star_search(start_coord, end_coord, self.grid, self.moves)