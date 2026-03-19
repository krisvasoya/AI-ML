import random

GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def manhattan_distance(state, goal=GOAL_STATE):
    distance = 0
    for i, val in enumerate(state):
        if val != 0:
            gi = goal.index(val)
            distance += abs(i // 3 - gi // 3) + abs(i % 3 - gi % 3)
    return distance

def misplaced_tiles(state, goal=GOAL_STATE):
    return sum(1 for i, val in enumerate(state) if val != 0 and val != goal[i])

def get_neighbors(state):
    b = state.index(0)
    moves = []
    if b % 3 > 0: moves.append(b - 1)
    if b % 3 < 2: moves.append(b + 1)
    if b > 2:     moves.append(b - 3)
    if b < 6:     moves.append(b + 3)
    
    neighbors = []
    for m in moves:
        # Check invalid row wrap
        if (m == b - 1 or m == b + 1) and (b // 3 != m // 3):
            continue
        new_state = list(state)
        new_state[b], new_state[m] = new_state[m], new_state[b]
        neighbors.append(tuple(new_state))
    return neighbors

def generate_random_puzzle(moves_count=20):
    state = list(GOAL_STATE)
    for _ in range(moves_count):
        b = state.index(0)
        moves = []
        if b % 3 > 0: moves.append(b - 1)
        if b % 3 < 2: moves.append(b + 1)
        if b > 2:     moves.append(b - 3)
        if b < 6:     moves.append(b + 3)
        m = random.choice(moves)
        state[b], state[m] = state[m], state[b]
    return tuple(state)
