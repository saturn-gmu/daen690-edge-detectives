# === scripts/run_ablation.py ===
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Project root

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score

from src.data.generate_features import get_or_generate_features
from src.config import config
from src.ablation.ablation_helper import evaluate_thresholded_feature_mask

# ----------------- Load and Preprocess Data ------------------
features_path = config.Paths.features_parquet
print("📦 Loading cached or raw features from:", features_path)

df, _ = get_or_generate_features(
    features_path=features_path,
    ranked_path=config.Paths.ranked_features_csv  # still used below to save ranking
)

feature_cols = [col for col in df.columns if col.startswith("f_")]
X = df[feature_cols].values
y = df["target"].values
print(f"🔢 Feature matrix shape: {X.shape}")

# ----------------- Train-Test Split --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# ----------------- Train Base Random Forest ------------------
print("🌲 Training base Random Forest...")
rf_base = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_base.fit(X_train, y_train)

# ----------------- Output Directories ------------------------
os.makedirs(config.Paths.ablation_threshold_masks, exist_ok=True)
os.makedirs(config.Paths.ablation_results, exist_ok=True)

# ----------------- Feature Importance Ranking ----------------
print("🧪 Running ablation across Top-5 to Top-64 features...")
importances = rf_base.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
sorted_features = [feature_cols[i] for i in sorted_indices]

# ✅ Save ranked features as column names like "f_77"
ranked_df = pd.DataFrame({"feature": sorted_features})
ranked_path = config.Paths.ranked_features_csv
ranked_path.parent.mkdir(parents=True, exist_ok=True)
ranked_df.to_csv(ranked_path, index=False)
print(f"📄 Saved ranked features to: {ranked_path}")

accuracies, precisions, features_retained = [], [], []

# ----------------- Main Ablation Loop ------------------------
for k in range(5, 65):
    print(f"🔬 Evaluating Top-{k} features")
    mask = np.zeros_like(importances, dtype=bool)
    mask[sorted_indices[:k]] = True

    # Save binary mask for reproducibility
    mask_path = config.Paths.ablation_threshold_masks / f"feature_mask_{k}.npy"
    np.save(mask_path, mask)

    # Evaluate with custom mask
    result = evaluate_thresholded_feature_mask(
        rf_base, X_train, X_test, y_train, y_test,
        threshold=None,
        is_bottleneck=False,
        custom_mask=mask
    )

    if result[0] is None:
        print("⚠️ Skipping due to insufficient features.")
        continue

    rf_masked, X_test_masked, _, _, metrics = result
    y_pred = rf_masked.predict(X_test_masked)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)

    print(f"✅ Top-{k} | Acc: {acc:.4f} | Prec: {prec:.4f} | F1: {metrics['F1']:.4f} | "
          f"MCC: {metrics['MCC']:.4f} | AUC: {metrics['AUC']:.4f} | AP: {metrics['Average Precision']:.4f}")

    accuracies.append(acc)
    precisions.append(prec)
    features_retained.append(k)

    joblib.dump(rf_masked, config.Paths.ablation_threshold_masks / f"rf_features_{k}.pkl")

# ----------------- Visualization -----------------------------
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
plot_path = config.Paths.ablation_results / f"ablation_accuracy_precision_{timestamp}.png"
plt.savefig(plot_path)
print(f"📈 Plot saved to {plot_path}")
plt.show()
