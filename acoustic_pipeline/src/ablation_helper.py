#ablation_helper.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier

def evaluate_thresholded_feature_mask(
    rf, X_train, X_test, y_train, y_test,
    threshold=0.01,
    is_bottleneck=False,
    custom_mask=None
):
    """
    Evaluates a Random Forest classifier using a masked subset of input features.
    """
    if custom_mask is not None:
        mask = custom_mask
    else:
        importances = rf.feature_importances_
        mask = importances >= threshold

    if np.sum(mask) <= 3:
        print("⚠️ Skipping: Too few features.")
        return None, None, mask

    X_train_masked = X_train[:, mask]
    X_test_masked = X_test[:, mask]

    rf_new = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    rf_new.fit(X_train_masked, y_train)

    return rf_new, X_test_masked, mask

