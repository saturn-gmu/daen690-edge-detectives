#clear_arifacts.py
# This script is used to clear all artifacts from previous runs of the pipeline.

import os
import glob
from pathlib import Path

# Define all artifact directories and matching file patterns to delete
cleanup_map = {
    "artifacts/default": [
        "*.npz", "*.npy", "*.pkl"
    ],
    "artifacts/quant_dnn": [
        "*.h5", "*.pkl", "*.npz"
    ],
    "artifacts/quant_top_features": [
        "*.h5", "*.pkl", "*.npz"
    ],
    "artifacts/tflite": [
        "*.tflite"
    ],
    "artifacts/tflite_ptq": [
        "*.tflite", "*.pkl", "*.h5"
    ],
    "artifacts/thresholded": [
    "feature_mask_*.npy",
    "rf_features_*.pkl",
    "rf_thresh_*.pkl"  # ← ADD THIS
    ],
"artifacts/thresholded": [
    "feature_mask_*.npy",
    "rf_features_*.pkl",
    "rf_thresh_*.pkl",
    "features_scaled.npz",
    "features_targets.pkl",
    "feature_scaler.pkl",
    "targets_all.pkl",
    "bottleneck_features_all.npy"
],
"artifacts/default": [
    "features_scaled.npz",
    "features_targets.pkl",
    "feature_scaler.pkl",
    "targets_all.pkl",
    "bottleneck_features_all.npy"
],

}

# Execute deletion
for folder, patterns in cleanup_map.items():
    for pattern in patterns:
        full_pattern = os.path.join(folder, pattern)
        for file_path in glob.glob(full_pattern):
            try:
                os.remove(file_path)
                print(f"🗑️ Deleted: {file_path}")
            except Exception as e:
                print(f"⚠️ Failed to delete {file_path}: {e}")

print("\n✅ All artifacts removed. Ready for a clean pipeline run.")
