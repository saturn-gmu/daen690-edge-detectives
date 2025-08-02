# ablation_helper.py
# This module provides a helper function to evaluate a Random Forest model
# using a thresholded or custom-selected subset of input features.

import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score
)

def evaluate_thresholded_feature_mask(
    rf,
    X_train,
    X_test,
    y_train,
    y_test,
    X_valid=None,
    y_valid=None,
    threshold=0.01,
    is_bottleneck=False,
    custom_mask=None
):
    """
    Evaluates a Random Forest classifier using a masked subset of input features.

    Parameters:
    - rf: Base RandomForestClassifier (used to obtain feature importances)
    - X_train, X_test: Feature matrices
    - y_train, y_test: Labels
    - X_valid, y_valid: Optional validation set
    - threshold: Importance cutoff (if no custom_mask)
    - is_bottleneck: Reserved
    - custom_mask: Boolean array for selecting specific features

    Returns:
    - rf_new: Trained RF model
    - X_test_masked: Test features with selected features
    - mask: Boolean feature mask
    - X_valid_masked: Masked validation set (or None)
    - metrics: Dictionary containing F1, MCC, AUC, and Average Precision
    """

    # Use custom mask if provided, otherwise threshold importances
    if custom_mask is not None:
        mask = custom_mask
    else:
        importances = rf.feature_importances_
        mask = importances >= threshold

    # Guard against too few features
    if np.sum(mask) <= 3:
        print("⚠️ Skipping: Too few features.")
        return None, None, mask, None, None

    # Apply mask to datasets
    X_train_masked = X_train[:, mask]
    X_test_masked = X_test[:, mask]
    X_valid_masked = X_valid[:, mask] if X_valid is not None else None

    # Train new RF on masked data
    rf_new = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    rf_new.fit(X_train_masked, y_train)

    # Predict and compute evaluation metrics
    y_pred = rf_new.predict(X_test_masked)
    y_scores = rf_new.predict_proba(X_test_masked)[:, 1]  # Probabilities for ROC/AUC

    metrics = {
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Average Precision": average_precision_score(y_test, y_scores),
        "AUC": roc_auc_score(y_test, y_scores)
    }

    return rf_new, X_test_masked, mask, X_valid_masked, metrics

def load_ranked_features(path="DNN_Model/artifacts/ablation/ranked_features.csv"):
    ranked_path = Path(path)
    if not ranked_path.exists():
        raise FileNotFoundError(f"❌ Ranked feature file not found at {ranked_path}")
    ranked = pd.read_csv(ranked_path)
    return ranked["feature"].tolist()