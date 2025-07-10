#!/usr/bin/env python
# DNN_Initial.py
# Trains a full-resolution DNN pipeline and evaluates it using metrics and saved plots.

import sys
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.dnn_model import train_dnn_model
from src.evaluation.evaluate_model import evaluate_model
from src.models.rf_model import train_random_forest, plot_cumulative_and_dynamic_importance
from src.utils.utils import start_timer, end_timer
from src.config.config import folder_path, vesselnames, vessel

# === Load or generate preprocessed dataset ===
FEATURES_PKL_PATH = Path("artifacts/default/features_df.pkl")

if FEATURES_PKL_PATH.exists():
    print(f"📦 Loading cached preprocessed dataset from {FEATURES_PKL_PATH}")
    df = pd.read_pickle(FEATURES_PKL_PATH)
else:
    print("🚧 Preprocessed features not found, running audio preprocessing pipeline...")
    from src.data.preprocess import load_data_and_extract_features
    df = load_data_and_extract_features(folder_path, vesselnames, vessel)

if df.empty:
    raise RuntimeError("❌ No data available for training. Check preprocessing and dataset integrity.")

print(f"✅ Loaded {len(df)} samples for model training.")

def main():
    total_start = start_timer()
    print("🚀 Running full-resolution DNN pipeline\n")

    # --- Load features and labels ---
    t0 = time.time()
    X = np.vstack(df['features'].values)
    y = df['target'].values
    print(f"📦 Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"    (load time: {time.time() - t0:.2f}s)\n")

    print("🔍 Label distribution:")
    print(df['target'].value_counts())

    # --- Train/test/validation split and scaling ---
    t1 = time.time()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.4, random_state=42)
    X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    X_valid_scaled = scaler.transform(X_valid)
    print(f"🔧 Split & scale complete (time: {time.time() - t1:.2f}s)\n")

    # --- Train DNN model ---
    t2 = time.time()
    model, history = train_dnn_model(X_train_scaled, y_train, X_valid_scaled, y_valid)
    print(f"🏋️  DNN training complete (time: {time.time() - t2:.2f}s)\n")

    # --- Evaluate DNN model ---
    t3 = time.time()
    y_scores = model.predict(X_test_scaled).flatten()
    y_pred   = (y_scores > 0.5).astype(int)

    evaluate_model(
        model=model,
        X_test=X_test_scaled,
        y_test=y_test,
        predicted_labels=y_pred,
        y_scores=y_scores,
        title="Full DNN",
        vessel=vessel
    )
    print(f"📊 DNN evaluation complete (time: {time.time() - t3:.2f}s)\n")

    # --- Save DNN model and scaler ---
    output_dir = Path("artifacts/default")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "dnn_model_fullres.h5"
    scaler_path = output_dir / "scaler_fullres.pkl"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"💾 Saved model to: {model_path}")
    print(f"💾 Saved scaler to: {scaler_path}\n")

    # --- Train and visualize Random Forest feature importance ---
    print("🔬 Training Random Forest on full feature set...")
    rf_model = train_random_forest(X_train_scaled, y_train)
    plot_cumulative_and_dynamic_importance(rf_model)

    # --- Done ---
    total_elapsed = end_timer(total_start)
    print(f"\n✅ Pipeline complete (total time: {total_elapsed:.2f}s)")

if __name__ == "__main__":
    main()
