import heapq
import numpy as np
from typing import List, Tuple, Optional

# Type Aliases for clarity
GridCoord = Tuple[int, int, int]
Heuristic = 'Heuristic' # Forward declaration for type hinting

def a_star_search(start: GridCoord, end: GridCoord, grid: np.ndarray, moves: List[GridCoord], heuristic: Heuristic) -> Optional[List[GridCoord]]:
    """
    A generic and fast A* search algorithm.

    This function is domain-agnostic. It only knows how to search a grid based
    on a provided heuristic. The "intelligence" comes from the heuristic object.

    Args:
        start: The starting coordinate on the grid.
        end: The target coordinate on the grid.
        grid: The 3D numpy array representing the environment (0=free, 1=obstacle).
        moves: A list of possible moves (e.g., all 26 directions in 3D).
        heuristic: An instantiated heuristic object with a `calculate(node, goal)` method.

    Returns:
        A list of grid coordinates representing the path, or None if no path is found.
    """
    open_set = [(0, start)]  # (f_score, node)
    came_from = {}
    g_score = {start: 0}
    open_set_hash = {start}

    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current not in open_set_hash: continue
        open_set_hash.remove(current)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for move in moves:
            neighbor = (current[0] + move[0], current[1] + move[1], current[2] + move[2])

            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1] and 0 <= neighbor[2] < grid.shape[2]):
                continue
            if grid[neighbor] == 1: continue

            # g_score is the simple, fast geometric distance.
            move_cost = np.linalg.norm(move)
            tentative_g_score = g_score.get(current, float('inf')) + move_cost
            
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                
                # The heuristic provides the intelligent, physics-aware estimate.
                h_score = heuristic.calculate(neighbor)
                f_score = tentative_g_score + h_score
                
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score, neighbor))
                    open_set_hash.add(neighbor)
                    
    return None # No path found