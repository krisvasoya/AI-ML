# AI Smart 8-Puzzle Solver

This project demonstrates AI Problem Solving & Search techniques applied to the 8-Puzzle problem, along with Machine Learning applications.

## Project Structure
- `main.py`: Main entry point orchestrating puzzle generation, solving, and ML training.
- `puzzle_solver.py`: Implements two search algorithms (BFS and A* with Manhattan distance heuristic).
- `ml_model.py`: Custom Linear Regression model trained using Gradient Descent to predict puzzle difficulty (number of steps to solve).
- `visualizer.py`: Uses `matplotlib` to chart performance comparisons between BFS and A* (Time and Nodes Explored).
- `utils.py`: Helper functions for the puzzle grid logic (Manhattan distance, generating valid neighbors).
- `data/`: Auto-generated CSV datasets for ML training.
- `results/`: Auto-generated comparison charts.

## Setup & Running
1. **Requirements:** `python 3.x`, `matplotlib`
   ```bash
   pip install matplotlib
   ```
   *(Note: Matplotlib is optional, but required to see `results/graphs.png`)*
   
2. **Execute:**
   ```bash
   python main.py
   ```

## Academic Mapping (Course Outcomes)
- **CO-1 (Search Algorithms):** BFS provides optimal but slow uninformed search (`O(b^d)`), while A* utilizes the Admissible Manhattan Heuristic for fast, informed, optimal search.
- **CO-4/5 (Machine Learning):** Real-time data generation solves hundreds of puzzles automatically to create dataset features (Manhattan distance, Misplaced tiles). A Linear Regression model is mapped to predict actual required steps to solution.
