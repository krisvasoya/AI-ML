import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Paths
DATA_PATH = "python-engine/data/puzzle_ml_dataset_700.csv"
MODEL_DIR = "python-engine/models/"
RESULTS_DIR = "python-engine/results/"

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_model():
    print("--- Loading and Preprocessing Data ---")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    # Features and Target
    # We choose features that describe the state, not the solver output
    features = [
        'manhattan_distance', 'misplaced_tiles', 'linear_conflict', 
        'corner_misplaced', 'blank_row', 'blank_col', 'blank_in_center', 
        'max_tile_displacement', 'num_valid_moves', 'scramble_moves'
    ]
    target = 'difficulty_label'

    X = df[features]
    y = df[target]

    # Handle missing values if any
    X = X.fillna(X.mean())

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Define difficulty order for better visualization if needed
    # trivial, easy, medium, hard, very_hard
    print(f"Classes: {label_encoder.classes_}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("--- Training RandomForestClassifier ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)

    print("--- Evaluating Model ---")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - Puzzle Difficulty Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {RESULTS_DIR}confusion_matrix.png")

    # Feature Importance Visualization
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importances.png'))
    print(f"Feature importance plot saved to {RESULTS_DIR}feature_importances.png")

    # Final training on all data
    print("--- Saving Model Artifacts ---")
    full_scaler = StandardScaler()
    X_scaled = full_scaler.fit_transform(X)
    model.fit(X_scaled, y_encoded)

    # Save artifacts
    artifacts = {
        'model': model,
        'scaler': full_scaler,
        'label_encoder': label_encoder,
        'feature_names': features
    }
    
    with open(os.path.join(MODEL_DIR, 'difficulty_model.pkl'), 'wb') as f:
        pickle.dump(artifacts, f)
    
    print(f"Model artifacts saved to {MODEL_DIR}difficulty_model.pkl")

if __name__ == "__main__":
    train_model()
