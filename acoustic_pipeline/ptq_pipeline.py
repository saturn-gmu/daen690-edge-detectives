#!/usr/bin/env python
# ptq_pipeline_patched.py
# This script performs Post-Training Quantization (PTQ) on a DNN model using TensorFlow Lite.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import os
import joblib
import tensorflow as tf
import time
from datetime import datetime 
import matplotlib.pyplot as plt
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow.lite")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.data.dataloader import load_data_and_extract_features
from src.config.config import folder_path, vesselnames, vessel
from src.models.dnn_model import build_consistent_dnn_model

# 🔧 Configurations
use_minmax_scaler = False
output_dir = "artifacts/tflite_ptq"

# 📦 Load and prepare dataset
print("📦 Loading dataset...")
df = load_data_and_extract_features(folder_path, vesselnames, vessel)

X_all = np.vstack(df['features'].values)
y_all = df['target'].values

# 📊 Track metrics
accuracies = []
precisions = []
feature_counts = []
timing_log = []
records = []

# 🔁 Loop over feature masks
for k in range(5, 65):
    feature_mask_path = f"artifacts/thresholded/feature_mask_{k}.npy"
    if not os.path.exists(feature_mask_path):
        continue

    start_time = time.time()
    print(f"\n🏁 Running quantization pipeline for Top-{k} features")

    mask = np.load(feature_mask_path)
    X_masked = X_all[:, mask]

    X_train, X_temp, y_train, y_temp = train_test_split(X_masked, y_all, stratify=y_all, test_size=0.4, random_state=42)
    X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)

    scaler = MinMaxScaler() if use_minmax_scaler else StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_valid_scaled = scaler.transform(X_valid)

    model = build_consistent_dnn_model(input_shape=(X_train_scaled.shape[1],))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, validation_data=(X_valid_scaled, y_valid), epochs=10, batch_size=64)

    loss, acc = model.evaluate(X_test_scaled, y_test)
    print(f"✅ Float32 model accuracy: {acc:.4f}")

    start_float = time.time()
    _ = model.predict(X_test_scaled, batch_size=64)
    end_float = time.time()
    float32_time_ms = (end_float - start_float) * 1000 / len(X_test_scaled)

    def representative_dataset():
        for row in X_train_scaled:
            yield [np.expand_dims(row, axis=0).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    scale, zp = input_details["quantization"]

    if scale == 0:
        raise ValueError("Quantization scale is zero. Check representative dataset and scaler.")

    X_test_int8 = ((X_test_scaled / scale) + zp).astype(np.int8)

    start_int8 = time.time()
    for x in X_test_int8:
        interpreter.set_tensor(input_details["index"], np.expand_dims(x, axis=0))
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])
    end_int8 = time.time()
    int8_time_ms = (end_int8 - start_int8) * 1000 / len(X_test_scaled)

    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, f"dnn_model_top{k}_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    model.save(os.path.join(output_dir, f"dnn_model_top{k}.h5"))
    joblib.dump(scaler, os.path.join(output_dir, f"scaler_top{k}.pkl"))

    os.makedirs("artifacts/quant_top_features", exist_ok=True)
    joblib.dump(scaler, f"artifacts/quant_top_features/scaler_top{k}.pkl")
    model.save(f"artifacts/quant_top_features/dnn_model_top{k}.h5")

    os.makedirs("artifacts/tflite", exist_ok=True)
    with open(f"artifacts/tflite/dnn_model_top{k}_int8.tflite", "wb") as f:
        f.write(tflite_model)

    print(f"✅ Quantized INT8 model for Top-{k} saved.")

    elapsed = time.time() - start_time
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"🕒 Total time for Top-{k}: {elapsed:.2f} seconds")
    print(f"📅 Completed Top-{k} at {timestamp} | Duration: {elapsed:.2f} seconds\n")

    accuracies.append(acc)

    from sklearn.metrics import precision_score
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
    prec = precision_score(y_test, y_pred)
    precisions.append(prec)
    feature_counts.append(k)

    records.append({
        "Model": f"Top-{k} TFLite",
        "Accuracy": acc,
        "Precision": prec,
        "Inference Time (ms)": int8_time_ms
    })

# 📈 Plotting and saving
if feature_counts:
    plt.figure(figsize=(10, 6))
    plt.plot(feature_counts, accuracies, marker='o', label='Accuracy')
    plt.plot(feature_counts, precisions, marker='s', label='Precision')
    plt.xlabel("Number of Features")
    plt.ylabel("Score")
    plt.title("INT8 Model Accuracy & Precision vs Feature Count")
    plt.grid(True)
    plt.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/ptq_accuracy_precision_vs_features.png")
    print("📈 Saved performance plot to results/ptq_accuracy_precision_vs_features.png")

if timing_log:
    df_timing = pd.DataFrame(timing_log)
    df_timing.to_csv("results/ptq_inference_comparison.csv", index=False)
    print("📊 Saved inference timing to results/ptq_inference_comparison.csv")

    plt.figure(figsize=(10, 6))
    plt.plot(df_timing["Top-K"], df_timing["Float32 (ms/sample)"], marker='o', label="Float32 DNN")
    plt.plot(df_timing["Top-K"], df_timing["INT8 (ms/sample)"], marker='s', label="TFLite INT8")
    plt.xlabel("Number of Features")
    plt.ylabel("Inference Time (ms per sample)")
    plt.title("Inference Time: Float32 vs INT8")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/ptq_inference_time_comparison.png")
    print("📈 Saved inference time plot to results/ptq_inference_time_comparison.png")

if records:
    df_out = pd.DataFrame(records)
    df_out.to_csv("results/pipeline_metrics.csv", index=False)
    print("📊 Saved performance metrics to results/pipeline_metrics.csv")
