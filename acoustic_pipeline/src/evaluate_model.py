# evaluate_model.py
# Evaluates a trained model and saves confusion matrix, ROC, and PR curves for dashboard use

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve, auc
)

def evaluate_model(model, X_test, y_test, predicted_labels=None, title="Model", vessel=None, y_scores=None):
    os.makedirs("results", exist_ok=True)

    # Get predictions and scores if not passed
    if y_scores is None:
        y_scores = model.predict(X_test).flatten()

    if predicted_labels is None:
        y_pred = (y_scores > 0.5).astype(int)
    else:
        y_pred = predicted_labels

    # Compute metrics with safe zero_division
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n📊 Evaluation Metrics for {title}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    prefix = vessel if vessel else title.lower().replace(" ", "_")
    label_suffix = f" ({vessel})" if vessel else ""

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {title}{label_suffix}")
    plt.savefig(f"results/confusion_matrix_{prefix}.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {title}{label_suffix}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/roc_curve_{prefix}.png")
    plt.close()

    # Precision-Recall Curve
    precs, recalls, _ = precision_recall_curve(y_test, y_scores)
    plt.figure()
    plt.plot(recalls, precs)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {title}{label_suffix}")
    plt.grid(True)
    plt.savefig(f"results/pr_curve_{prefix}.png")
    plt.close()
