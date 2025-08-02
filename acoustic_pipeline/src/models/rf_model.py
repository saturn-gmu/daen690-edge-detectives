from pathlib import Path

# Define the updated code content (same as the one user provided and confirmed)
# rf_model.py
# This file provides Random Forest training and analysis tools:
# - Training a balanced RF classifier
# - Plotting cumulative feature importance
# - Generating evaluation plots: confusion matrix, ROC curve, and PR curve

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve, auc
)
from pathlib import Path

def train_random_forest(X_train, y_train, random_state=42):
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=random_state
    )
    rf.fit(X_train, y_train)
    return rf

def plot_cumulative_and_dynamic_importance(model, save_path=None):
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    cumulative = np.cumsum(sorted_importances)

    plt.figure(figsize=(10, 5))
    plt.plot(cumulative, marker='o', label="Cumulative Importance")
    plt.axhline(y=0.95, color='r', linestyle='--', label="95% Threshold")
    plt.xlabel("Top-K Features")
    plt.ylabel("Cumulative Importance")
    plt.title("Cumulative Feature Importance (Random Forest)")
    plt.grid(True)
    plt.legend()

    save_path = Path("DNN_Model/artifacts/results/plots/feature_importance_rf.png") if save_path is None else Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_model_evaluation_curves(y_true, y_pred, y_scores=None, prefix="rf"):
    output_dir = Path("DNN_Model/artifacts/results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix: {prefix.upper()}")
    plt.savefig(output_dir / f"confusion_matrix_{prefix}.png")
    plt.close()

    if y_scores is None:
        raise ValueError("y_scores must be provided to plot ROC curve")
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {prefix.upper()}")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / f"roc_curve_{prefix}.png")
    plt.close()

    prec, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {prefix.upper()}")
    plt.grid(True)
    plt.savefig(output_dir / f"pr_curve_{prefix}.png")
    plt.close()

