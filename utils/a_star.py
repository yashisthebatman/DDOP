# utils/a_star.py
import heapq
import numpy as np
from typing import List, Tuple, Optional, Callable

# Type Aliases for clarity
GridCoord = Tuple[int, int, int]
Heuristic = 'Heuristic' # Forward declaration for type hinting

def a_star_search(
    start: GridCoord, 
    end: GridCoord, 
    grid: np.ndarray, 
    moves: List[GridCoord], 
    heuristic: Heuristic or Callable
) -> Optional[List[GridCoord]]:
    """
    A generic A* search algorithm. Now expects the heuristic object to also provide
    the move cost, making the search metric-aware (time, energy, etc.).
    For simple cases (like the coarse search), a basic lambda function can be used.
    """
    open_set = [(0, start)]  # (f_score, node)
    came_from = {}
    g_score = {start: 0}
    open_set_hash = {start}
    
    is_heuristic_object = hasattr(heuristic, 'calculate')

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

            # If we have a full heuristic object, use it to get the true cost of the move.
            # Otherwise, use simple geometric distance (for the coarse search).
            if is_heuristic_object:
                p_prev = came_from.get(current)
                move_cost = heuristic.cost_between(current, neighbor, p_prev)
            else: # Simple Euclidean distance for coarse search
                move_cost = np.linalg.norm(move)

            tentative_g_score = g_score.get(current, float('inf')) + move_cost
            
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                
                # The heuristic provides the intelligent, physics-aware estimate to the goal.
                if is_heuristic_object:
                    h_score = heuristic.calculate(neighbor)
                else: # Simple Euclidean distance heuristic for coarse search
                    h_score = heuristic(neighbor, end)

                f_score = tentative_g_score + h_score
                
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score, neighbor))
                    open_set_hash.add(neighbor)
                    
    return None # No path found