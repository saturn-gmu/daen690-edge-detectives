# scripts/dnn_summary.py

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from joblib import load as joblib_load
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, roc_auc_score,
    f1_score, roc_curve, precision_recall_curve, auc
)

# Add parent directory to Python path to import local modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config

# ---------------------- USER PROMPT ------------------------
print("\n💡 Choose benchmarking mode:")
print("1. Simple (TFLite only)")
print("2. Comprehensive (TFLite, DNN, RF)")
mode = input("Enter 1 or 2: ").strip()
is_comprehensive = mode == "2"
print(f"\n🔧 Running in {'comprehensive' if is_comprehensive else 'simple'} mode\n")

# ---------------------- LOAD DATA --------------------------
print("📦 Loading features...")
df = pd.read_parquet(config.Paths.features_parquet)

# Handle legacy vs expanded format
if 'features' in df.columns:
    X_all = np.vstack(df['features'].values)
else:
    feature_cols = [col for col in df.columns if col.startswith("f_")]
    X_all = df[feature_cols].values

y_all = df["target"].values
print(f"✅ Loaded dataset: {X_all.shape}, Labels: {y_all.shape}")

# ---------------------- OUTPUT SETUP -----------------------
os.makedirs(config.Paths.pipeline_results, exist_ok=True)
os.makedirs(config.Paths.pipeline_plots, exist_ok=True)
records = []

# ---------------------- TIMER FUNCTION ---------------------
def timeit(label, func, *args, **kwargs):
    print(f"⏱️ {label}...")
    start = time.time()
    result = func(*args, **kwargs)
    duration = time.time() - start
    print(f"✅ {label} completed in {duration:.3f}s")
    return result, duration

# ---------------------- MAIN EVALUATION LOOP ---------------
def evaluate_k(k):
    try:
        print(f"\n🔍 Evaluating Top-{k} features")
        mask_path = config.Paths.feature_masks / f"feature_mask_{k}.npy"
        if not mask_path.exists():
            print(f"⚠️ Missing mask for Top-{k}, skipping.")
            return

        mask = np.load(mask_path)
        X_k = X_all[:, mask]

        # ------------------ TFLite INT8 Evaluation ------------------
        tflite_path = config.Paths.tflite_models_dir / f"dnn_model_top{k}_int8.tflite"
        if tflite_path.exists():
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]

            scale, zp = input_details['quantization']
            X_q = ((X_k / scale) + zp).astype(np.int8)

            def tflite_pred():
                preds, scores = [], []
                for x in X_q:
                    interpreter.set_tensor(input_details["index"], np.expand_dims(x, axis=0))
                    interpreter.invoke()
                    output = interpreter.get_tensor(output_details["index"])
                    prob = output[0][0]
                    preds.append(int(prob > 0.5))
                    scores.append(prob)
                return np.array(preds), np.array(scores)

            (y_pred_tflite, tflite_scores), t_tflite = timeit("TFLite inference", tflite_pred)

            records.append({
                "Model": f"Top-{k} TFLite",
                "Top-K": k,
                "Accuracy": accuracy_score(y_all, y_pred_tflite),
                "Precision": precision_score(y_all, y_pred_tflite),
                "AUC": roc_auc_score(y_all, tflite_scores),
                "F1 Score": f1_score(y_all, y_pred_tflite),
                "Inference Time (ms)": t_tflite * 1000
            })
            plot_roc_pr(y_all, tflite_scores, k, "TFLite")

        # ------------------ DNN + RF Evaluation ------------------
        if is_comprehensive:
            dnn_path = config.Paths.h5_models_dir / f"dnn_model_top{k}.h5"
            if dnn_path.exists():
                model = load_model(dnn_path)

                def dnn_pred():
                    scores = model.predict(X_k, batch_size=64).flatten()
                    return (scores > 0.5).astype(int), scores

                (y_pred_dnn, dnn_scores), t_dnn = timeit("DNN inference", dnn_pred)

                records.append({
                    "Model": f"Top-{k} DNN",
                    "Top-K": k,
                    "Accuracy": accuracy_score(y_all, y_pred_dnn),
                    "Precision": precision_score(y_all, y_pred_dnn),
                    "AUC": roc_auc_score(y_all, dnn_scores),
                    "F1 Score": f1_score(y_all, y_pred_dnn),
                    "Inference Time (ms)": t_dnn * 1000
                })
                plot_roc_pr(y_all, dnn_scores, k, "DNN")

            rf_path = config.Paths.feature_masks / f"rf_features_{k}.pkl"
            if rf_path.exists():
                rf = joblib_load(rf_path)
                X_rf = X_k[:, :rf.n_features_in_]

                def rf_pred():
                    return rf.predict(X_rf), rf.predict_proba(X_rf)[:, 1]

                (y_pred_rf, rf_scores), t_rf = timeit("Random Forest inference", rf_pred)

                records.append({
                    "Model": f"Top-{k} RF",
                    "Top-K": k,
                    "Accuracy": accuracy_score(y_all, y_pred_rf),
                    "Precision": precision_score(y_all, y_pred_rf),
                    "AUC": roc_auc_score(y_all, rf_scores),
                    "F1 Score": f1_score(y_all, y_pred_rf),
                    "Inference Time (ms)": t_rf * 1000
                })
                plot_roc_pr(y_all, rf_scores, k, "RF")

    except Exception as e:
        print(f"❌ Error @ Top-{k}: {e}")

# ------------------ PLOTTING FUNCTION ------------------
def plot_roc_pr(y_true, scores, k, model_label):
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    auc_roc = auc(fpr, tpr)
    auc_pr = auc(rec, prec)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_roc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC Curve ({model_label} Top-{k})")
    plt.legend()
    plt.savefig(config.Paths.pipeline_plots / f"roc_top{k}_{model_label}.png")
    plt.close()

    plt.figure()
    plt.plot(rec, prec, label=f"PR AUC = {auc_pr:.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR Curve ({model_label} Top-{k})")
    plt.legend()
    plt.savefig(config.Paths.pipeline_plots / f"pr_top{k}_{model_label}.png")
    plt.close()

# ------------------ MAIN LOOP -------------------------
for k in range(5, 65):
    evaluate_k(k)

# ------------------ SAVE FINAL RESULTS ----------------
df_results = pd.DataFrame(records)
df_results.to_csv(config.Paths.pipeline_results / "pipeline_metrics.csv", index=False)
print("\n✅ Metrics saved to pipeline_metrics.csv")

# ------------------ PLOT INFERENCE TIME ---------------
if not df_results.empty:
    df_sorted = df_results[df_results.Model.str.contains("TFLite|DNN")].sort_values("Top-K")
    plt.figure(figsize=(10, 6))
    for label in ["DNN", "TFLite"]:
        subset = df_sorted[df_sorted.Model.str.contains(label)]
        plt.plot(subset["Top-K"], subset["Inference Time (ms)"], marker='o', label=label)

    plt.xlabel("Top-K Features")
    plt.ylabel("Inference Time (ms)")
    plt.title("Inference Time: DNN vs TFLite")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(config.Paths.pipeline_results / "inference_time_comparison.png")
    print("📈 Saved inference time plot")
2