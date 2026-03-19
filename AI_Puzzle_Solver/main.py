import os
import sys

# Fix Windows console encoding to support UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from utils import generate_random_puzzle, GOAL_STATE, manhattan_distance, misplaced_tiles
from puzzle_solver import bfs, astar
from ml_model import LinearRegressionModel, load_training_data_from_csv
from visualizer import plot_algorithm_comparison

DATASET_PATH = "data/training_data.csv"
WEIGHTS_PATH = "data/model_weights.csv"

OK   = "[OK]"
FAIL = "[FAIL]"

def print_board(state):
    for i in range(0, 9, 3):
        row = f" {state[i]} | {state[i+1]} | {state[i+2]}"
        print(row.replace("0", " "))
    print()

def section(title, num):
    print(f"\n{'='*52}")
    print(f"  [{num}] {title}")
    print(f"{'='*52}")

# ─────────────────────────────────────────────
# 1. Generate puzzle
# ─────────────────────────────────────────────
section("Generating a solvable 8-puzzle", 1)
start_state = generate_random_puzzle(moves_count=20)
print("Initial State:")
print_board(start_state)
print("Goal State:")
print_board(GOAL_STATE)

# ─────────────────────────────────────────────
# 2. BFS
# ─────────────────────────────────────────────
section("Solving with Breadth-First Search (BFS)", 2)
bfs_result = bfs(start_state, GOAL_STATE)
if "fail" not in bfs_result:
    print(f"  {OK} Solved in {bfs_result['steps']} steps")
    print(f"     Nodes explored : {bfs_result['nodes']}")
    print(f"     Time taken     : {bfs_result['time']:.2f} ms")
else:
    print(f"  {FAIL} BFS timed out or failed")

# ─────────────────────────────────────────────
# 3. A*
# ─────────────────────────────────────────────
section("Solving with A* (Manhattan Distance Heuristic)", 3)
astar_result = astar(start_state, GOAL_STATE)
if "fail" not in astar_result:
    print(f"  {OK} Solved in {astar_result['steps']} steps")
    print(f"     Nodes explored : {astar_result['nodes']}")
    print(f"     Time taken     : {astar_result['time']:.2f} ms")
else:
    print(f"  {FAIL} A* timed out or failed")

# ─────────────────────────────────────────────
# 4. Visualization
# ─────────────────────────────────────────────
section("Generating Performance Graphs", 4)
if "fail" not in bfs_result and "fail" not in astar_result:
    plot_algorithm_comparison(
        bfs_result["nodes"], astar_result["nodes"],
        bfs_result["time"],  astar_result["time"],
    )
else:
    print("  Skipping graph -- one algorithm failed.")

# ─────────────────────────────────────────────
# 5. ML -- Load large dataset + train
# ─────────────────────────────────────────────
section("Training ML Model on Large Dataset", 5)

model = LinearRegressionModel()

# Try to load pre-saved weights first (fast start on repeated runs)
loaded = model.load(WEIGHTS_PATH)

if not loaded:
    print(f"  Loading dataset from {DATASET_PATH} ...")
    training_data = load_training_data_from_csv(DATASET_PATH)

    if not training_data:
        print("  No dataset found. Generating 200 samples on the fly...")
        from ml_model import generate_training_data
        training_data = generate_training_data(200, DATASET_PATH)

    print(f"  Dataset size : {len(training_data)} samples")
    print("  Training Linear Regression model (Gradient Descent) ...")
    success = model.train(training_data)

    if success:
        model.save(WEIGHTS_PATH)
        print(f"  {OK} Training complete!")

        if model.train_loss_history:
            last_ep, last_mse = model.train_loss_history[-1]
            print(f"     Final MSE  (epoch {last_ep}) : {last_mse:.4f}")

        mae, rmse = model.evaluate(training_data)
        print(f"     MAE  : {mae:.2f} steps")
        print(f"     RMSE : {rmse:.2f} steps")
        print(f"\n  Learned weights:")
        print(f"     w0 (bias)      = {model.w0:.4f}")
        print(f"     w1 (Manhattan) = {model.w1:.4f}")
        print(f"     w2 (Misplaced) = {model.w2:.4f}")
    else:
        print(f"  {FAIL} Training failed -- not enough data.")

# ─────────────────────────────────────────────
# 6. Prediction on the current puzzle
# ─────────────────────────────────────────────
section("ML Prediction for Current Puzzle", 6)
if model.trained:
    m_dist    = manhattan_distance(start_state, GOAL_STATE)
    m_tiles   = misplaced_tiles(start_state, GOAL_STATE)
    predicted = model.predict(m_dist, m_tiles)
    actual    = astar_result.get("steps", "N/A")

    print(f"  Features for current puzzle:")
    print(f"     Manhattan Distance : {m_dist}")
    print(f"     Misplaced Tiles    : {m_tiles}")
    print(f"\n  Actual steps needed  : {actual}")
    print(f"  ML Predicted steps   : {predicted}")

    if actual != "N/A":
        error_steps = abs(actual - predicted)
        print(f"  Prediction error     : {error_steps} steps")
else:
    print("  Model not trained -- no prediction available.")

# ─────────────────────────────────────────────
print(f"\n{'='*52}")
print("  DONE")
print(f"  Dataset  : {DATASET_PATH}")
print(f"  Weights  : {WEIGHTS_PATH}")
print(f"  Graph    : results/graphs.png")
print(f"{'='*52}\n")
