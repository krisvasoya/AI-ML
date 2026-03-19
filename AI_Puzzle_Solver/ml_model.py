import csv
import os
import random
from utils import GOAL_STATE, manhattan_distance, misplaced_tiles, generate_random_puzzle
from puzzle_solver import astar


class LinearRegressionModel:
    """
    Multi-variable Linear Regression trained with Gradient Descent.
    Features: Manhattan Distance (x1), Misplaced Tiles (x2)
    Target:   Actual steps to solve (y)
    """

    def __init__(self):
        self.w0 = 0.0   # intercept / bias
        self.w1 = 0.0   # weight for Manhattan distance
        self.w2 = 0.0   # weight for Misplaced tiles
        self.trained = False
        self.train_loss_history = []

    # ------------------------------------------------------------------
    # Feature scaling helpers (min-max normalisation stored after fit)
    # ------------------------------------------------------------------
    def _normalize(self, data):
        """Normalise features so gradient descent converges faster."""
        x1s = [d[0] for d in data]
        x2s = [d[1] for d in data]
        ys  = [d[2] for d in data]

        self.x1_min, self.x1_max = min(x1s), max(x1s)
        self.x2_min, self.x2_max = min(x2s), max(x2s)
        self.y_min,  self.y_max  = min(ys),  max(ys)

        def scale(v, lo, hi):
            return (v - lo) / (hi - lo + 1e-9)

        normalised = [
            (scale(x1, self.x1_min, self.x1_max),
             scale(x2, self.x2_min, self.x2_max),
             scale(y,  self.y_min,  self.y_max))
            for x1, x2, y in data
        ]
        return normalised

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, data):
        """
        Gradient descent on MSE loss.
        Works on raw (un-normalised) features for direct interpretability,
        but uses per-epoch shuffling for robustness.
        """
        m = len(data)
        if m < 5:
            print("Not enough data to train the model.")
            return False

        lr     = 0.0001
        epochs = 3000

        self.train_loss_history = []

        for ep in range(epochs):
            # Shuffle each epoch to avoid local minima
            random.shuffle(data)

            dw0 = dw1 = dw2 = 0.0
            total_loss = 0.0

            for x1, x2, y in data:
                pred = self.w0 + self.w1 * x1 + self.w2 * x2
                err  = pred - y
                total_loss += err ** 2
                dw0 += err
                dw1 += err * x1
                dw2 += err * x2

            self.w0 -= lr * dw0 / m
            self.w1 -= lr * dw1 / m
            self.w2 -= lr * dw2 / m

            # Record MSE every 100 epochs
            if (ep + 1) % 100 == 0:
                mse = total_loss / m
                self.train_loss_history.append((ep + 1, mse))

        self.trained = True
        return True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, x1, x2):
        if not self.trained:
            return 0
        raw = self.w0 + self.w1 * x1 + self.w2 * x2
        return max(0, round(raw))

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def evaluate(self, data):
        """Return MAE and RMSE over given data."""
        if not self.trained or not data:
            return None, None
        abs_errors = []
        sq_errors  = []
        for x1, x2, y in data:
            pred = self.w0 + self.w1 * x1 + self.w2 * x2
            abs_errors.append(abs(pred - y))
            sq_errors.append((pred - y) ** 2)
        mae  = sum(abs_errors) / len(abs_errors)
        rmse = (sum(sq_errors) / len(sq_errors)) ** 0.5
        return mae, rmse

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------
    def save(self, path="data/model_weights.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["w0", "w1", "w2"])
            writer.writerow([self.w0, self.w1, self.w2])
        print(f"  Model weights saved to {path}")

    def load(self, path="data/model_weights.csv"):
        if not os.path.exists(path):
            return False
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.w0 = float(row["w0"])
                self.w1 = float(row["w1"])
                self.w2 = float(row["w2"])
        self.trained = True
        print(f"  Loaded model weights from {path}")
        return True


# ======================================================================
# Dataset utilities
# ======================================================================

def load_training_data_from_csv(path="data/training_data.csv"):
    """Load existing dataset from CSV and return list of (x1, x2, y) tuples."""
    if not os.path.exists(path):
        return []

    data = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x1 = float(row["Manhattan"])
                x2 = float(row["Misplaced"])
                y  = float(row["Steps"])
                data.append((x1, x2, y))
            except (ValueError, KeyError):
                continue
    return data


def generate_training_data(num_samples=100, csv_path="data/training_data.csv",
                           append=False):
    """
    Generate `num_samples` new puzzles, solve each with A*, and save to CSV.
    If append=True, rows are appended; otherwise the file is overwritten.
    Returns list of (x1, x2, y) tuples that were newly generated.
    """
    data = []
    print(f"  Generating {num_samples} puzzles...")

    for i in range(num_samples):
        moves = random.randint(10, 35)
        state = generate_random_puzzle(moves)

        x1 = manhattan_distance(state, GOAL_STATE)
        x2 = misplaced_tiles(state, GOAL_STATE)

        res = astar(state, GOAL_STATE)
        if "fail" not in res:
            y = res["steps"]
            data.append((x1, x2, y))

        if (i + 1) % 500 == 0:
            print(f"    ... {i+1}/{num_samples} done")

    # Save / append
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    mode = "a" if append else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.writer(f)
        if not append:
            writer.writerow(["Manhattan", "Misplaced", "Steps"])
        writer.writerows(data)

    print(f"  {len(data)} new samples saved → {csv_path}")
    return data
