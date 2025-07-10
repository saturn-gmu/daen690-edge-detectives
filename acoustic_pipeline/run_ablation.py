#run_ablation.py 
# This script performs an ablation study on a Random Forest model by evaluating the impact of different feature subsets. 

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Ensure access to src/ modules

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score
import matplotlib.pyplot as plt
import datetime

from src.data.dataloader import load_data_and_extract_features
from src.config.config import folder_path, vesselnames, vessel
from src.ablation.ablation_helper import evaluate_thresholded_feature_mask

# --- Load fresh data and extract features (84-bin CQT expected) ---
print("📦 Loading data from folder:", folder_path)
df = load_data_and_extract_features(folder_path, vesselnames, vessel)
X = np.vstack(df['features'].values)
y = df['target'].values
print(f"🔢 Feature shape: {X.shape}")

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# --- Train base Random Forest ---
print("🌲 Training base Random Forest...")
rf_base = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_base.fit(X_train, y_train)

# --- Prepare output folders ---
os.makedirs("artifacts/thresholded", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- Perform ablation loop over Top-k features ---
print("🧪 Running ablation across Top-5 to Top-64 features...")
importances = rf_base.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

accuracies = []
precisions = []
features_retained = []

for k in range(5, 65):
    print(f"🔬 Evaluating Top-{k} features")
    mask = np.zeros_like(importances, dtype=bool)
    mask[sorted_indices[:k]] = True

    # Save the feature mask for downstream use
    np.save(f"artifacts/thresholded/feature_mask_{k}.npy", mask)

    # Evaluate and record performance
    result = evaluate_thresholded_feature_mask(
        rf_base, X_train, X_test, y_train, y_test,
        threshold=None,
        is_bottleneck=False,
        custom_mask=mask
    )

    if result[0] is None:
        continue

    rf_masked, X_test_masked, _ = result
    y_pred = rf_masked.predict(X_test_masked)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f} | Precision: {prec:.4f}")

    accuracies.append(acc)
    precisions.append(prec)
    features_retained.append(k)

    # Save masked RF model
    joblib.dump(rf_masked, f"artifacts/thresholded/rf_features_{k}.pkl")

# --- Plot and save results ---
plt.figure(figsize=(10, 6))
plt.plot(features_retained, accuracies, marker='o', label='Accuracy')
plt.plot(features_retained, precisions, marker='s', label='Precision')
plt.xlabel("Number of Features")
plt.ylabel("Score")
plt.title("Random Forest Accuracy & Precision vs Feature Count")
plt.grid(True)
plt.legend()
plt.tight_layout()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = f"results/ablation_accuracy_precision_{timestamp}.png"
plt.savefig(plot_path)
print(f"📈 Plot saved to {plot_path}")
plt.show()
