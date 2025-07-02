import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from src.ablation_helper import evaluate_thresholded_feature_mask
from src.evaluation import evaluate_model
from src.config import vessel
from sklearn.metrics import precision_score, accuracy_score
import datetime

# --- Load 64-D bottleneck features ---
data = np.load("artifacts/default/bottleneck_split.npz")
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# --- Ensure output directory exists ---
os.makedirs("artifacts/thresholded", exist_ok=True)

# --- Thresholds to evaluate ---
thresholds = np.arange(0.01, 0.031, 0.001)
accuracies = []
precisions = []
features_retained = []

# --- Loop through each threshold and run ablation ---
for t in thresholds:
    print("\n============================")
    print(f"🔬 Evaluating threshold: {t:.3f}")

    # Load full bottleneck RF model (64 features)
    rf = joblib.load("artifacts/default/rf_model.pkl")

    # Run ablation for bottleneck features
    rf_masked, X_test_masked, mask = evaluate_thresholded_feature_mask(
        rf, X_train, X_test, y_train, y_test, threshold=t, is_bottleneck=True
    )

    # Predict and evaluate manually (no confusion matrix)
    y_pred = rf_masked.predict(X_test_masked)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f}")

    num_features = np.sum(mask)
    accuracies.append(acc)
    precisions.append(prec)
    features_retained.append(num_features)

    # Save model and feature mask
    model_path = f"artifacts/thresholded/rf_thresh_{int(t * 1000)}.pkl"
    mask_path = f"artifacts/thresholded/feature_mask_{int(t * 1000)}.npy"
    joblib.dump(rf_masked, model_path)
    np.save(mask_path, mask)

    # Save individual models for 18–22 features
    if 18 <= num_features <= 22:
        joblib.dump(rf_masked, f"artifacts/thresholded/rf_features_{num_features}.pkl")
        np.save(f"artifacts/thresholded/feature_mask_{num_features}.npy", mask)

# --- Plot Accuracy and Precision vs. Number of Features Retained ---
plt.figure(figsize=(10, 6))
plt.plot(features_retained, accuracies, label="Accuracy", marker='o')
plt.plot(features_retained, precisions, label="Precision", marker='s')
plt.xlabel("Number of Features Retained")
plt.ylabel("Score")
plt.title("Ablation: Accuracy and Precision vs. Feature Count")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save timestamped version of the plot
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"results/ablation_accuracy_precision_{timestamp}.png"
plt.savefig(filename)
print(f"✅ Ablation results plot saved as: {filename}")
plt.show()