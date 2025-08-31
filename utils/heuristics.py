import heapq
import numpy as np

def _manhattan_distance(a, b):
    """A simple heuristic for grid-based movement."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

def a_star_search(start, end, grid, moves, node_subset=None, heuristic_func=None):
    """
    Finds a path from start to end using a standard, robust, and optimized A* algorithm.
    """
    heuristic = heuristic_func if heuristic_func else _manhattan_distance

    open_set = [(0, start)]  # Priority queue (min-heap)
    came_from = {}
    
    g_score = {start: 0}
    
    open_set_hash = {start} # For O(1) membership checking

    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current not in open_set_hash:
            continue # Already processed
        open_set_hash.remove(current)

        if current == end:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for move in moves:
            neighbor = (current[0] + move[0], current[1] + move[1], current[2] + move[2])

            # Boundary checks
            if not (0 <= neighbor[0] < grid.shape[0] and
                    0 <= neighbor[1] < grid.shape[1] and
                    0 <= neighbor[2] < grid.shape[2]):
                continue
            
            # Obstacle check
            if grid[neighbor[0], neighbor[1], neighbor[2]] == 1:
                continue

            tentative_g_score = g_score.get(current, float('inf')) + np.linalg.norm(move)
            
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, end)
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score, neighbor))
                    open_set_hash.add(neighbor)

    return None # Explicitly return None if no path is found