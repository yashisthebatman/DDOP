import heapq
from itertools import product
import numpy as np
import logging

from config import MAX_PATH_LENGTH
from typing import TYPE_CHECKING, Set, List, Tuple

if TYPE_CHECKING:
    from utils.coordinate_manager import CoordinateManager

class JumpPointSearch:
    def __init__(self, start, goal, is_obstructed_func, heuristic, coord_manager: "CoordinateManager"):
        self.start = start
        self.goal = goal
        self.is_obstructed = is_obstructed_func
        self.heuristic = heuristic
        self.coord_manager = coord_manager
        self.open_set = []
        self.open_set_map = {}
        self.came_from = {}

    def search(self):
        self.open_set = []
        self.open_set_map = {}
        self.came_from = {}
        try:
            if not self.coord_manager.is_valid_local_grid_pos(self.start):
                logging.error(f"JPS: Start position out of bounds or invalid: {self.start}")
                return None
            if not self.coord_manager.is_valid_local_grid_pos(self.goal):
                logging.error(f"JPS: Goal position out of bounds or invalid: {self.goal}")
                return None
            heapq.heappush(self.open_set, (0, self.start))
            self.open_set_map[self.start] = 0

            iterations = 0
            max_iterations = 100000
            while self.open_set and iterations < max_iterations:
                _, current = heapq.heappop(self.open_set)
                iterations += 1
                if current == self.goal:
                    path = self._reconstruct_path_grid(current)
                    # FIX: If reconstruction fails, it returns None. This must be handled
                    # to prevent the search from continuing with invalid data and entering an infinite loop.
                    if path is None:
                        logging.error("JPS path reconstruction failed, aborting search.")
                        return None
                        
                    if self.validate_full_path(path):
                        return path
                    else:
                        logging.warning("JPS: Invalid path found; failed validation.")
                        return None

                successors = self._get_successors(current)
                for successor in successors:
                    if not self.coord_manager.is_valid_local_grid_pos(successor):
                        continue
                    if successor not in self.open_set_map:
                        f_score = self.heuristic(successor)
                        heapq.heappush(self.open_set, (f_score, successor))
                        self.open_set_map[successor] = f_score
                    self.came_from[successor] = current

            if iterations >= max_iterations:
                logging.error("JPS: Max iterations exceeded. Possible infinite loop or unreachable goal.")
            else:
                logging.error("JPS: Failed to find a path.")
            return None
        except Exception as e:
            logging.exception(f"JPS crashed: {e}")
            return None

    def _reconstruct_path_grid(self, end):
        path = []
        current = end
        while current in self.came_from:
            prev = self.came_from[current]
            segment = self._get_line_cells(prev, current)
            valid_segment = self.validate_path_segment(segment)
            if not valid_segment:
                logging.error(f"JPS: Invalid path segment from {prev} to {current}")
                return None
            if path:
                path = segment[:-1] + path
            else:
                path = segment
            current = prev
        if current == self.start:
            if not path or path[0] != self.start:
                path = [self.start] + path
        return path

    def _get_line_cells(self, start, end):
        x0, y0, z0 = start; x1, y1, z1 = end
        points = []
        dx, dy, dz = abs(x1-x0), abs(y1-y0), abs(z1-z0)
        xs, ys, zs = (1 if x1 > x0 else -1), (1 if y1 > y0 else -1), (1 if z1 > z0 else -1)
        x, y, z = x0, y0, z0
        if dx >= dy and dx >= dz:
            p1, p2 = 2 * dy - dx, 2 * dz - dx
            for _ in range(dx + 1):
                points.append((x, y, z)); x += xs
                if p1 >= 0: y += ys; p1 -= 2 * dx
                if p2 >= 0: z += zs; p2 -= 2 * dx
                p1 += 2 * dy; p2 += 2 * dz
        elif dy >= dx and dy >= dz:
            p1, p2 = 2 * dx - dy, 2 * dz - dy
            for _ in range(dy + 1):
                points.append((x, y, z)); y += ys
                if p1 >= 0: x += xs; p1 -= 2 * dy
                if p2 >= 0: z += zs; p2 -= 2 * dy
                p1 += 2 * dx; p2 += 2 * dz
        else:
            p1, p2 = 2 * dy - dz, 2 * dx - dz
            for _ in range(dz + 1):
                points.append((x, y, z)); z += zs
                if p1 >= 0: y += ys; p1 -= 2 * dz
                if p2 >= 0: x += xs; p2 -= 2 * dz
                p1 += 2 * dy; p2 += 2 * dx
        return points

    def validate_path_segment(self, segment):
        for pt in segment:
            if not self.coord_manager.is_valid_local_grid_pos(pt):
                logging.warning(f"JPS: Grid position out of bounds: {pt}"); return False
            if self.is_obstructed(pt):
                logging.warning(f"JPS: Path cell obstructed: {pt}"); return False
        return True

    def validate_full_path(self, path):
        if path is None: return False
        for i in range(1, len(path)):
            segment = self._get_line_cells(path[i-1], path[i])
            if not self.validate_path_segment(segment): return False
        return True

    def _get_successors(self, node):
        neighbors = []
        for dx, dy, dz in product([-1, 0, 1], repeat=3):
            if (dx, dy, dz) == (0, 0, 0): continue
            neighbor = (node[0] + dx, node[1] + dy, node[2] + dz)
            if not self.coord_manager.is_valid_local_grid_pos(neighbor): continue
            if self.is_obstructed(neighbor): continue
            if self._has_forced_neighbor(node, (dx, dy, dz)): neighbors.append(neighbor)
            elif abs(dx) + abs(dy) + abs(dz) > 1: neighbors.append(neighbor)
        return neighbors

    def _has_forced_neighbor(self, node, direction):
        dx, dy, dz = direction; x, y, z = node; forced = False
        if dx != 0 and dy != 0:
            if self.is_obstructed((x, y+dy, z)) and not self.is_obstructed((x+dx, y+dy, z)): forced = True
            if self.is_obstructed((x+dx, y, z)) and not self.is_obstructed((x+dx, y+dy, z)): forced = True
        if dx != 0 and dz != 0:
            if self.is_obstructed((x, y, z+dz)) and not self.is_obstructed((x+dx, y, z+dz)): forced = True
            if self.is_obstructed((x+dx, y, z)) and not self.is_obstructed((x+dx, y, z+dz)): forced = True
        if dy != 0 and dz != 0:
            if self.is_obstructed((x, y+dy, z)) and not self.is_obstructed((x, y+dy, z+dz)): forced = True
            if self.is_obstructed((x, y, z+dz)) and not self.is_obstructed((x, y+dy, z+dz)): forced = True
        if dx != 0 and dy != 0 and dz != 0:
            faces, diagonal = [(x+dx, y, z), (x, y+dy, z), (x, y, z+dz)], (x+dx, y+dy, z+dz)
            if any(self.is_obstructed(face) for face in faces) and not self.is_obstructed(diagonal): forced = True
        return forced