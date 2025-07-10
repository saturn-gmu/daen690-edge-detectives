# Update the enhanced PTQ evaluation script to include special evaluation logic for Top-64
# evaluate_tflite_model_with_float.py
# Evaluates TFLite INT8 models against Float32 versions, with special Top-64 evaluation.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import entropy, pearsonr
from scipy.special import softmax
import tensorflow as tf
import joblib

from src.data.dataloader import load_data_and_extract_features
from src.config.config import folder_path, vesselnames, vessel

os.makedirs("results/plots", exist_ok=True)
df = load_data_and_extract_features(folder_path, vesselnames, vessel)
X_all = np.vstack(df['features'].values)
y_all = df['target'].values

def compute_ptq_metrics(float_scores, quant_scores, y_true):
    float_probs = softmax(np.vstack([1 - float_scores, float_scores]).T, axis=1)
    quant_probs = softmax(np.vstack([1 - quant_scores, quant_scores]).T, axis=1)
    delta_acc = accuracy_score(y_true, (quant_scores > 0.5)) - accuracy_score(y_true, (float_scores > 0.5))
    kl = np.mean([entropy(p, q) for p, q in zip(float_probs, quant_probs)])
    top1_match = np.mean((quant_scores > 0.5) == (float_scores > 0.5))
    corr, _ = pearsonr(float_scores, quant_scores)
    return delta_acc, kl, top1_match, corr

def run_tflite_inference(interpreter, X_test_scaled):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    scale, zero_point = input_details[0]['quantization']
    X_test_q = ((X_test_scaled / scale) + zero_point).astype(np.int8)
    y_pred, y_scores = [], []
    for x in X_test_q:
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(x, axis=0))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0][0]
        y_scores.append(output)
        y_pred.append(int(output > 0.5))
    return np.array(y_pred), np.array(y_scores)

metrics = []
for k in range(5, 65):
    try:
        print(f"🔍 Evaluating Top-{k} features...")
        mask = np.load(f"artifacts/thresholded/feature_mask_{k}.npy")
        scaler = joblib.load(f"artifacts/quant_top_features/scaler_top{k}.pkl")
        model_float = tf.keras.models.load_model(f"artifacts/quant_top_features/dnn_model_top{k}.h5")
        tflite_path = f"artifacts/tflite/dnn_model_top{k}_int8.tflite"

        X_masked = X_all[:, mask]
        X_train, X_temp, y_train, y_temp = train_test_split(X_masked, y_all, stratify=y_all, test_size=0.4, random_state=42)
        X_test, _, y_test, _ = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)
        X_test_scaled = scaler.transform(X_test)

        float_scores = model_float.predict(X_test_scaled).flatten()
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        start = time.time()
        y_pred_q, y_scores_q = run_tflite_inference(interpreter, X_test_scaled)
        int8_time = time.time() - start

        acc = accuracy_score(y_test, y_pred_q)
        prec = precision_score(y_test, y_pred_q)
        rec = recall_score(y_test, y_pred_q)
        f1 = f1_score(y_test, y_pred_q)
        auc = roc_auc_score(y_test, y_scores_q)
        d_acc, kl, top1, corr = compute_ptq_metrics(float_scores, y_scores_q, y_test)

        # Special logic for Top-64: save confusion matrix and curves
        if k == 64:
            cm = confusion_matrix(y_test, y_pred_q)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix: Top-64 INT8")
            plt.savefig("results/plots/confusion_matrix_top64.png", bbox_inches="tight")
            plt.close()

            fpr, tpr, _ = roc_curve(y_test, y_scores_q)
            plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
            plt.title("ROC Curve: Top-64 INT8")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.savefig("results/plots/roc_top64.png")
            plt.close()

        metrics.append({
            "Top-K": k,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AUC": auc,
            "ΔAccuracy": d_acc,
            "KL Divergence": kl,
            "Top-1 Agreement": top1,
            "Output Corr": corr,
            "Inference Time (ms)": int8_time * 1000
        })
    except Exception as e:
        print(f"⚠️ Skipping Top-{k}: {e}")

df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv("results/pipeline_metrics_extended.csv", index=False)

plt.figure()
plt.plot(df_metrics["Top-K"], df_metrics["Accuracy"], label="Accuracy")
plt.plot(df_metrics["Top-K"], df_metrics["AUC"], label="AUC")
plt.xlabel("Top-K Features")
plt.ylabel("Score")
plt.title("Accuracy & AUC vs Feature Count")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/accuracy_auc_vs_features.png")

plt.figure()
plt.plot(df_metrics["Top-K"], df_metrics["Inference Time (ms)"], label="Inference Time")
plt.xlabel("Top-K Features")
plt.ylabel("ms")
plt.title("Inference Time vs Feature Count")
plt.grid(True)
plt.savefig("results/plots/inference_time_vs_features.png")

print("✅ Evaluation complete. Results saved to CSV and plots.")

