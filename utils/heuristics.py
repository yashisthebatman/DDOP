# utils/heuristics.py
import heapq

def a_star_search(start, end, grid, moves):
    """
    Finds a path from start to end using the A* algorithm.
    'grid' is a 3D numpy array where 1 represents an obstacle.
    'moves' is a list of possible (dx, dy, dz) tuples.
    """
    def heuristic(a, b):
        # Manhattan distance heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while open_set:
        _, current = heapq.heappop(open_set)

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

            # Check if neighbor is valid
            if not (0 <= neighbor[0] < grid.shape[0] and
                    0 <= neighbor[1] < grid.shape[1] and
                    0 <= neighbor[2] < grid.shape[2]):
                continue
            if grid[neighbor[0], neighbor[1], neighbor[2]] == 1:
                continue

            tentative_g_score = g_score[current] + 1 # Assume cost of 1 for each step

            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None # No path found