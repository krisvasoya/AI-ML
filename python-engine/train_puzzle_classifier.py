import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Paths (adjusted to be relative to engine directory)
DATA_PATH = "data/puzzle_ml_dataset_700.csv"
MODEL_DIR = "models/"
RESULTS_DIR = "results/"

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
    print(f"Classes: {label_encoder.classes_}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("--- Evaluating Multiple Models ---")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Support Vector Machine (SVM)": SVC(probability=True, random_state=42)
    }

    best_model = None
    best_accuracy = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name}: Accuracy = {acc:.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_name = name
            best_model = model
            
    print(f"\n--- Best Model Selection: {best_name} ---")
    y_pred = best_model.predict(X_test_scaled)
    print(f"Accuracy: {best_accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {best_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {RESULTS_DIR}confusion_matrix.png")

    # Final training on all data
    print("--- Saving Model Artifacts ---")
    full_scaler = StandardScaler()
    X_scaled = full_scaler.fit_transform(X)
    best_model.fit(X_scaled, y_encoded)

    # Save artifacts
    artifacts = {
        'model': best_model,
        'scaler': full_scaler,
        'label_encoder': label_encoder,
        'feature_names': features,
        'model_name': best_name
    }
    
    with open(os.path.join(MODEL_DIR, 'difficulty_model.pkl'), 'wb') as f:
        pickle.dump(artifacts, f)
    
    print(f"Model artifacts saved to {MODEL_DIR}difficulty_model.pkl")

if __name__ == "__main__":
    train_model()
