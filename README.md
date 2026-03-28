# AI Puzzle Solver & Machine Learning Visualizer

This project is an interactive, analytical platform designed to solve the classic 8-Puzzle game and analyze algorithmic performance using both classical search strategies (BFS, A*) and predictive Machine Learning classifiers.

## 🚀 Overview
The platform generates solvable 8-puzzles, visually tracks the solution algorithms, directly compares the performance footprints (nodes explored, time taken), and utilizes a trained Machine Learning model to evaluate a puzzle's inherent difficulty class (such as Trivial, Medium, or Very Hard) exclusively from structural heuristics. 

---

## 🛠️ Technical Information & Tech Stack

### Frontend Architecture
- **Structure**: Vanilla **HTML5** & **CSS3** (Glassmorphism layout, dynamic keyframe animations, CSS variables, custom typography including *Syne* and *Space Mono*).
- **Interactivity**: Vanilla **JavaScript** (`src/script.js`) powering the graph search calculations, solution playback engine, and ML prediction simulation.
- **Visualizations**: Integrated **Chart.js** locally for tracking logic step comparisons and batch analysis curves. **Lucide Icons** implemented for scalable crisp iconography.

### Machine Learning / Backend Engine
- **Language**: **Python 3.x**
- **Libraries Utilized**:
  - `pandas` / `numpy`: Data manipulation, handling the 700-sample dataset matrix operations.
  - `scikit-learn`: Core ML library used for model building (`LogisticRegression`, `RandomForestClassifier`), data preprocessing (`StandardScaler`, `LabelEncoder`), dataset splitting (`train_test_split`), and performance evaluation (`accuracy_score`, `confusion_matrix`, `classification_report`).
  - `matplotlib` / `seaborn`: Rendering diagnostic graphs like the confusion matrix heatmap.
  - `pickle`: Serializing and saving the trained model architecture for potential future use.

---

## 🧠 Algorithms, Logic & Key Functions

### 1. Breadth-First Search (BFS)
An uninformed exhaustive search approach. BFS systematically checks every possible branch of puzzle moves layer by layer.
- **JS Function**: `bfs(start, goal)` inside `src/script.js`
- **Logic**: Uses a standard array as a `Queue`. It generates neighboring valid states (up, down, left, right) and adds them to the queue if not previously visited. It guarantees finding the shortest optimal path but consumes massive amounts of memory tracking all branches.

### 2. A* Search (A-Star)
An informed, highly optimized graph search minimizing path cost.
- **JS Function**: `astar(start, goal)` inside `src/script.js`
- **Logic**: Driven by the cost formula `f(n) = g(n) + h(n)` (where `n` is the node, `g` is actual steps taken so far, and `h` is estimated heuristic cost left). It uses a sorted open list (acting as a priority queue/min-heap) to continually explore the most promising paths first.
- **Heuristic Functions**:
  - `manhattan(state, goal)`: Calculates the sum of absolute grid coordinate distances each tile needs to move to reach its goal position.
  - `misplacedTiles(state, goal)`: A simpler heuristic returning the raw count of tiles sitting in the wrong destination slot.

### 3. Machine Learning Predictor (Logistic Regression Classifier)
Instead of forcing a brute-force solve to determine puzzle difficulty, the AI engine relies on a pre-trained ML classifier.
- **Python Function**: `train_model()` inside `python-engine/train_puzzle_classifier.py`
- **Logic**: 
  - The script iterates through candidate models (Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machines).
  - Multi-class Logistic Regression with pseudo-Newton optimization (L-BFGS) emerged as the best performing model.
  - Using scaled standardized variables (`StandardScaler`), the model transforms inputs like `manhattan_distance`, `linear_conflict`, and `num_valid_moves` into a categorical difficulty estimation.
  - **JS Counterpart**: `updatePrediction()` in `src/script.js` maps real-time UI sliders to visually simulate these predicted difficulty classes directly for the user on the web interface.

### 4. Interactive Utilities
- **JS Function**: `runSolver()` - Orchestrates UI loading states, executes the selected algorithm logic, and paints result variables into the HTML DOM.
- **JS Function**: `animateValue(elem, start, end, duration)` - Provides satisfying, smooth number-counting visual flair when solver metrics are printed.
- **JS Function**: `stepSolution(dir)` & `startPlay()` - Engine loop controlling the automated sequence playback of the found path on the UI grid.

---

## 📁 Repository Structure
```text
/
├── index.html                   # Main User Interface + DOM Elements
├── src/
│   ├── style.css                # Polished interactive animations and styles
│   └── script.js                # Core puzzle graph logic + predictive UI
├── python-engine/
│   ├── data/
│   │   └── puzzle_ml_dataset_700.csv # ML feature set
│   ├── models/                  # Exported .pkl pipeline weights & scalars
│   ├── results/                 # Exported Python metric visualizations
│   └── train_puzzle_classifier.py    # Master machine learning pipeline script
└── README.md
```

---

## 🔧 How to Run Locally

### Simulating the Website
Because the website uses relative directory paths, simply launching a localized web server provides the best experience so CORS policies don't complain about chart generation:
1. Ensure Python 3 is installed.
2. Open your terminal at the core folder directory.
3. Run the following generic HTTP command:
   ```bash
   python -m http.server 3000
   ```
4. Access the portal at: `http://localhost:3000`

### Re-training the ML Model
If you ever want to iterate or retrain the predictive engine natively on your local machine:
1. Navigate to the `/python-engine` folder.
2. Run standard python interpreter:
   ```bash
   python train_puzzle_classifier.py
   ```
3. Check the console for metrics, and the `/results` folder for updated graphical visualizations of the training loops!
