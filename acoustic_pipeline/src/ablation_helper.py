# ablation_helper.py
# Updated to handle both full and bottleneck features safely

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import joblib


def evaluate_thresholded_feature_mask(rf, X_train, X_test, y_train, y_test, threshold=0.01, is_bottleneck=False):
    importances = rf.feature_importances_
    mask = importances >= threshold

    # For bottleneck features, use mask directly
    if is_bottleneck:
        X_train_masked = X_train[:, mask]
        X_test_masked = X_test[:, mask]
        print(f"✅ Bottleneck features retained: {np.sum(mask)} / {len(mask)}")
    else:
        # Expect input to be CQT-derived features (mean + std)
        # Separate mask into mean/std for 84+84 inputs
        if len(mask) != 64:
            raise ValueError("Non-bottleneck masking only supported with 64D RF feature importances.")

        full_mask = np.concatenate([mask, mask])
        if full_mask.shape[0] != X_train.shape[1]:
            raise IndexError(f"Mask length mismatch: expected {X_train.shape[1]}, got {full_mask.shape[0]}")
        X_train_masked = X_train[:, full_mask]
        X_test_masked = X_test[:, full_mask]
        print(f"✅ Full features retained: {np.sum(full_mask)} / {len(full_mask)}")

    rf_new = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_new.fit(X_train_masked, y_train)
    y_pred = rf_new.predict(X_test_masked)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    print(f"✅ Retrained RF Accuracy: {acc:.4f}, Precision: {prec:.4f}")

    return rf_new, X_test_masked, mask
