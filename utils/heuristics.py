# utils/heuristics.py
import heapq
import numpy as np

def _manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

def a_star_search(start, end, grid, moves, node_subset=None, heuristic_func=None):
    """
    Finds a path from start to end using the A* algorithm.
    'heuristic_func' (optional) is a custom function h(current, end) for guiding the search.
    """
    heuristic = heuristic_func if heuristic_func else _manhattan_distance

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for move in moves:
            neighbor = (current[0] + move[0], current[1] + move[1], current[2] + move[2])

            if not (0 <= neighbor[0] < grid.shape[0] and
                    0 <= neighbor[1] < grid.shape[1] and
                    0 <= neighbor[2] < grid.shape[2]):
                continue
            
            if grid[neighbor[0], neighbor[1], neighbor[2]] == 1:
                continue

            if node_subset is not None and neighbor not in node_subset:
                continue

            tentative_g_score = g_score[current] + np.linalg.norm(move)
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None