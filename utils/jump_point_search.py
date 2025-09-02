# ==============================================================================
# File: utils/jump_point_search.py (Definitive, Corrected Version)
# ==============================================================================
import heapq
from itertools import product
import numpy as np
import logging

from config import MAX_PATH_LENGTH
from typing import TYPE_CHECKING, Set, List, Tuple

if TYPE_CHECKING:
    from utils.coordinate_manager import CoordinateManager

class JumpPointSearch:
    def __init__(self, start: tuple, goal: tuple, is_obstructed_func: callable, heuristic, coord_manager: 'CoordinateManager'):
        self.start = start
        self.goal = goal
        self.is_obstructed = is_obstructed_func
        self.heuristic = heuristic
        self.coord_manager = coord_manager
        self.open_set = []
        self.open_set_map = {}
        self.came_from = {}
        self.g_score = {start: 0}
        self.directions = list(product([-1, 0, 1], repeat=3))
        self.directions.remove((0, 0, 0))
        self.MAX_JUMP_DEPTH = MAX_PATH_LENGTH
        self.max_iterations = 20000

    def search(self):
        if self.start == self.goal:
            return [self.start]
        h_start = self.heuristic.calculate(self.start)
        heapq.heappush(self.open_set, (h_start, self.start))
        self.open_set_map[self.start] = h_start
        iterations = 0
        while self.open_set and iterations < self.max_iterations:
            iterations += 1
            _, current = heapq.heappop(self.open_set)
            if current == self.goal:
                return self._reconstruct_path_grid(current)
            if current in self.open_set_map:
                del self.open_set_map[current]
            successors = self._identify_successors(current)
            for successor in successors:
                new_g_score = self.g_score[current] + np.linalg.norm(np.array(current) - np.array(successor))
                if successor not in self.g_score or new_g_score < self.g_score.get(successor, float('inf')):
                    self.g_score[successor] = new_g_score
                    f_score = new_g_score + self.heuristic.calculate(successor)
                    if successor not in self.open_set_map:
                        heapq.heappush(self.open_set, (f_score, successor))
                        self.open_set_map[successor] = f_score
                    self.came_from[successor] = current
        logging.warning("JPS search failed to find a path or hit max iterations.")
        return None

    def _identify_successors(self, node):
        successors = set()
        parent = self.came_from.get(node)
        if parent is None:
            pruned_directions = self.directions
        else:
            direction = tuple(np.sign(np.array(node) - np.array(parent)).astype(int))
            pruned_directions = self._prune_directions(node, direction)
        for d in pruned_directions:
            jump_point = self._jump(node, d, set())
            if jump_point:
                successors.add(jump_point)
        return list(successors)

    def _jump(self, node: tuple, direction: tuple, visited: set):
        next_node = (node[0] + direction[0], node[1] + direction[1], node[2] + direction[2])

        if not self.coord_manager.is_valid_grid_position(next_node) or self.is_obstructed(next_node) or next_node in visited:
            return None
        
        visited.add(next_node)

        if next_node == self.goal:
            return next_node

        if self._has_forced_neighbor(next_node, direction):
            return next_node
            
        dx, dy, dz = direction
        # Diagonal move case
        if dx != 0 and dy != 0 or dx != 0 and dz != 0 or dy != 0 and dz != 0:
            if self._jump(next_node, (dx, 0, 0), visited.copy()):
                return next_node
            if self._jump(next_node, (0, dy, 0), visited.copy()):
                return next_node
            if self._jump(next_node, (0, 0, dz), visited.copy()):
                return next_node
        
        # Continue searching in the same direction
        return self._jump(next_node, direction, visited)

    def _has_forced_neighbor(self, node, direction):
        dx, dy, dz = direction
        x, y, z = node
        
        # Only diagonal moves can have forced neighbors
        if len([i for i in direction if i != 0]) < 2:
            return False

        # 2D Diagonal Checks
        if dx != 0 and dy != 0 and dz == 0:
            if self.is_obstructed((x - dx, y, z)) and not self.is_obstructed((x - dx, y + dy, z)): return True
            if self.is_obstructed((x, y - dy, z)) and not self.is_obstructed((x + dx, y - dy, z)): return True
        elif dx != 0 and dz != 0 and dy == 0:
            if self.is_obstructed((x - dx, y, z)) and not self.is_obstructed((x - dx, y, z + dz)): return True
            if self.is_obstructed((x, y, z - dz)) and not self.is_obstructed((x + dx, y, z - dz)): return True
        elif dy != 0 and dz != 0 and dx == 0:
            if self.is_obstructed((x, y - dy, z)) and not self.is_obstructed((x, y - dy, z + dz)): return True
            if self.is_obstructed((x, y, z - dz)) and not self.is_obstructed((x, y + dy, z - dz)): return True
        # 3D Diagonal Checks
        elif dx != 0 and dy != 0 and dz != 0:
            if self._has_forced_neighbor(node, (dx, dy, 0)): return True
            if self._has_forced_neighbor(node, (dx, 0, dz)): return True
            if self._has_forced_neighbor(node, (0, dy, dz)): return True

        return False

    def _prune_directions(self, node: tuple, direction: tuple) -> List[Tuple[int, int, int]]:
        pruned_dirs = []
        dx, dy, dz = direction
        
        # If last move was straight
        if len([i for i in direction if i != 0]) == 1:
            pruned_dirs.append(direction) # Always continue in the same direction
            # Add forced neighbors
            if dx != 0: # Moving along X
                if self.is_obstructed((node[0], node[1] + 1, node[2])) and not self.is_obstructed((node[0] - dx, node[1] + 1, node[2])): pruned_dirs.append((dx, 1, 0))
                if self.is_obstructed((node[0], node[1] - 1, node[2])) and not self.is_obstructed((node[0] - dx, node[1] - 1, node[2])): pruned_dirs.append((dx, -1, 0))
            elif dy != 0: # Moving along Y
                if self.is_obstructed((node[0] + 1, node[1], node[2])) and not self.is_obstructed((node[0] + 1, node[1] - dy, node[2])): pruned_dirs.append((1, dy, 0))
                if self.is_obstructed((node[0] - 1, node[1], node[2])) and not self.is_obstructed((node[0] - 1, node[1] - dy, node[2])): pruned_dirs.append((-1, dy, 0))
        # If last move was diagonal
        else:
            # Natural neighbors
            if dx != 0: pruned_dirs.append((dx, 0, 0))
            if dy != 0: pruned_dirs.append((0, dy, 0))
            if dz != 0: pruned_dirs.append((0, 0, dz))
            # Forced neighbors
            if self._has_forced_neighbor(node, direction):
                pruned_dirs.append(direction)
        
        return list(set(pruned_dirs))

    def _reconstruct_path_grid(self, current):
        total_path = [current]
        node = current
        while node in self.came_from:
            parent = self.came_from[node]
            segment = self._get_line_cells(parent, node)
            total_path = segment[:-1] + total_path
            node = parent
        return total_path

    def _get_line_cells(self, p1, p2):
        p1_arr, p2_arr = np.array(p1), np.array(p2)
        dist = np.linalg.norm(p2_arr - p1_arr)
        num_steps = int(np.ceil(dist))
        if num_steps == 0: return [p1]
        path = []
        for i in range(num_steps + 1):
            t = i / num_steps
            point = tuple(np.round(p1_arr * (1 - t) + p2_arr * t).astype(int))
            if not path or point != path[-1]: path.append(point)
        return path