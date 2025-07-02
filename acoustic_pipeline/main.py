# This script runs the default full-resolution pipeline using bottleneck features for Random Forest ablation or
# Runs the default or top-20 bottleneck DNN pipeline with optional timing breakdown

from src.dataloader import load_data_and_extract_features
from src.dnn_model import train_dnn_model, extract_bottleneck_features
from src.evaluation import evaluate_model, plot_history
from src.rf_model import train_random_forest, plot_cumulative_and_dynamic_importance
from src.utils import start_timer, end_timer
from src.config import folder_path, vesselnames, vessel

import numpy as np
import joblib
import os
import time
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Toggle to use top-20 bottleneck pipeline
use_top20 = True

# Start total timer
total_start = start_timer()

print("\U0001F680 Running", "top-20 bottleneck pipeline" if use_top20 else "full-resolution DNN pipeline")

# --- Load data ---
data_start = time.time()
df = load_data_and_extract_features(folder_path, vesselnames, vessel)
X_full = np.vstack(df['features'].values)
y = df['target'].values
data_end = time.time()

if use_top20:
    # --- Load top 20 mask and bottleneck model ---
    mask = np.load("artifacts/thresholded/feature_mask_20.npy")
    model_full = load_model("artifacts/quant_dnn/dnn_model.h5")
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_full)
    X_train_b, X_test_b, X_all_b = extract_bottleneck_features(model_full, X_full_scaled, X_full_scaled, scaler_full, df)
    X = X_all_b[:, mask]
else:
    X = X_full

# --- Split & scale ---
split_start = time.time()
X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.4, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_valid_scaled = scaler.transform(X_valid)
split_end = time.time()

# --- Train model ---
train_start = time.time()
model, history = train_dnn_model(X_train_scaled, y_train, X_valid_scaled, y_valid)
train_end = time.time()

plot_history(history)

# --- Evaluate ---
eval_start = time.time()
evaluate_model(model, X_test_scaled, y_test, title="Top 20 DNN" if use_top20 else "Full DNN", vessel=vessel)
eval_end = time.time()

# --- Train RF on bottleneck or full ---
print("\n\U0001F52C Training Random Forest on extracted features...")
rf_model = train_random_forest(X_train_scaled, y_train)
plot_cumulative_and_dynamic_importance(rf_model)

# --- Report timing ---
print(f"\n⏱️ Data loading time: {data_end - data_start:.2f} seconds")
print(f"⏱️ Split & scale time: {split_end - split_start:.2f} seconds")
print(f"⏱️ Training time: {train_end - train_start:.2f} seconds")
print(f"⏱️ Evaluation time: {eval_end - eval_start:.2f} seconds")

end_timer(total_start)
print("✅ Pipeline execution complete")