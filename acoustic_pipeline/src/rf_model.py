# rf_model.py
# This file contains functions to plot cumulative and dynamic feature importance for a Random Forest model,
# as well as model evaluation curves like confusion matrix, ROC, and precision-recall curves.
# It assumes the model has been trained and the necessary libraries are installed.

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve, auc
)

def train_random_forest(X_train, y_train, random_state=42):
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state)
    rf.fit(X_train, y_train)
    return rf

def plot_cumulative_and_dynamic_importance(model, save_path="results/feature_importance_rf.png"):
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
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_model_evaluation_curves(y_true, y_pred, y_scores=None, prefix="rf"):
    os.makedirs("results", exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix: {prefix.upper()}")
    plt.savefig(f"results/confusion_matrix_{prefix}.png")
    plt.close()

    # ROC Curve
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
    plt.savefig(f"results/roc_curve_{prefix}.png")
    plt.close()

    # PR Curve
    prec, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {prefix.upper()}")
    plt.grid(True)
    plt.savefig(f"results/pr_curve_{prefix}.png")
    plt.close()

# Example call in RF evaluation pipeline
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score

    # Dummy data for testing integration (replace with actual preprocessed dataset)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, size=100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = train_random_forest(X_train, y_train)

    y_scores = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)

    # Save evaluation plots
    plot_model_evaluation_curves(y_test, y_pred, y_scores, prefix="rf")
    plot_cumulative_and_dynamic_importance(rf, save_path="results/feature_importance_rf.png")
