import time
import heapq
from collections import deque
from utils import get_neighbors, manhattan_distance, GOAL_STATE

def bfs(start_state, goal_state=GOAL_STATE):
    start_time = time.time()
    if start_state == goal_state:
        return {'path': [start_state], 'steps': 0, 'nodes': 1, 'time': 0}
    
    queue = deque([ (start_state, [start_state]) ])
    visited = set([start_state])
    nodes = 0
    
    while queue:
        state, path = queue.popleft()
        nodes += 1
        
        for neighbor in get_neighbors(state):
            if neighbor not in visited:
                new_path = path + [neighbor]
                if neighbor == goal_state:
                    time_taken = (time.time() - start_time) * 1000
                    return {'path': new_path, 'steps': len(new_path)-1, 'nodes': nodes, 'time': time_taken}
                visited.add(neighbor)
                queue.append((neighbor, new_path))
        
        if nodes > 100000:  # Timeout safety
            break

    return {'path': [], 'steps': -1, 'nodes': nodes, 'time': (time.time() - start_time) * 1000, 'fail': True}

def astar(start_state, goal_state=GOAL_STATE):
    start_time = time.time()
    if start_state == goal_state:
        return {'path': [start_state], 'steps': 0, 'nodes': 1, 'time': 0}
    
    open_set = []
    # (f_score, tie_breaker, state, path, g_score)
    heapq.heappush(open_set, (manhattan_distance(start_state), 0, start_state, [start_state], 0))
    visited = set()
    nodes = 0
    tie_breaker = 1
    
    while open_set:
        f, _, state, path, g = heapq.heappop(open_set)
        
        if state in visited:
            continue
            
        visited.add(state)
        nodes += 1
        
        if state == goal_state:
            time_taken = (time.time() - start_time) * 1000
            return {'path': path, 'steps': len(path)-1, 'nodes': nodes, 'time': time_taken}
            
        for neighbor in get_neighbors(state):
            if neighbor not in visited:
                new_g = g + 1
                new_f = new_g + manhattan_distance(neighbor, goal_state)
                new_path = path + [neighbor]
                heapq.heappush(open_set, (new_f, tie_breaker, neighbor, new_path, new_g))
                tie_breaker += 1
                
        if nodes > 100000:
            break
            
    return {'path': [], 'steps': -1, 'nodes': nodes, 'time': (time.time() - start_time) * 1000, 'fail': True}
