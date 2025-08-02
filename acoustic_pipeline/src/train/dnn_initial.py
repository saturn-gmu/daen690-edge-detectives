# DNN_Initial.py
# === Top-K DNN Initial Benchmarking with Float32 metric naming + .h5 export + .npz rep data export ===

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings

import argparse
from pathlib import Path
import sys
import time
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score
from sklearn.dummy import DummyClassifier

import tensorflow as tf
from tensorflow.keras import backend as K

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.data.generate_features import get_or_generate_features
from src.models.dnn_model import DNNClassifier
from src.evaluation.evaluate_model import (
    evaluate_predictions,
    plot_training_history,
    plot_precision_recall_roc,
    plot_feature_importances,
    plot_test_vs_validation_metrics
)
from src.utils.utils import split_and_scale
from src.quantization.tflite_converter import convert_model_to_tflite

def run_dnn_topk(df, ranked_features, out_dir, topk_start=config.TopK.start, topk_end=config.TopK.end, topk_step=config.TopK.step):
    results = []
    feature_log_dir = config.Paths.dnn_feature_logs
    plot_dir = config.Paths.dnn_plots
    feature_log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    num_classes = df["target"].nunique()

    for top_k in range(topk_start, topk_end, topk_step):
        print(f"\n🔁 Training DNN on Top-{top_k} features...")
        selected = ranked_features[:top_k]
        selected = [f for f in selected if f in df.columns]
        if not selected:
            print(f"⚠️ Skipping Top-{top_k} — no valid features.")
            continue

        with open(feature_log_dir / f"top_{top_k:02d}.txt", "w") as f:
            f.write("\n".join(selected))

        X = df[selected]
        y = df["target"]

        scaler, splits = split_and_scale(X, y)
        X_train, X_test, X_valid = splits["X_train"], splits["X_test"], splits["X_valid"]
        y_train, y_test, y_valid = splits["y_train"], splits["y_test"], splits["y_valid"]

        # Baseline dummy classifier
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        y_dummy = dummy.predict(X_test)
        baseline_accuracy = accuracy_score(y_test, y_dummy)
        baseline_precision = precision_score(y_test, y_dummy, zero_division=0)
        print(f"🔹 Baseline Accuracy:  {baseline_accuracy:.4f}")
        print(f"🔹 Baseline Precision: {baseline_precision:.4f}")

        # Train DNN
        with tf.device("/CPU:0"):
            K.clear_session()
            model = DNNClassifier(input_shape=(X_train.shape[1],), num_classes=num_classes)
            history = model.fit(X_train, y_train, X_valid, y_valid)

        plot_training_history(history, plot_dir / f"training_top_{top_k}.png")

        # Inference and timing
        start_inf = time.time()
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)
        elapsed_inf = round(time.time() - start_inf, 4)

        # Save .h5 model
        h5_path = config.Paths.h5_models_dir / f"dnn_top_{top_k}.h5"
        model.model.save(h5_path)
        print(f"📅 Saved Keras model to {h5_path}")

        # Export TFLite Float32
        tflite_path = config.Paths.tflite_models_dir / f"float32_top_{top_k}.tflite"
        convert_model_to_tflite(model.model, tflite_path, quantize=False)
        model_size_kb = round(tflite_path.stat().st_size / 1024, 2)

        # Export representative dataset for quantization
        rep_out_path = config.Paths.representative_data / f"rep_data_top_{top_k}.npz"
        np.savez_compressed(rep_out_path, X=X_train)
        print(f"📦 Saved representative dataset to {rep_out_path}")

        # Test metrics
        base_metrics = evaluate_predictions(y_test, y_pred, y_scores)
        metrics = {f"{k} (Float32)": v for k, v in base_metrics.items()}
        metrics["MCC (Float32)"] = matthews_corrcoef(y_test, y_pred)
        metrics["Elapsed (s)"] = elapsed_inf
        metrics["Model Size (KB)"] = model_size_kb

        print("\n📈 Post-training Evaluation:")
        print(f"✅ Top-{top_k} | Accuracy ↑ {metrics['Accuracy (Float32)']:.4f} vs {baseline_accuracy:.4f} | "
              f"Precision ↑ {metrics['Precision (Float32)']:.4f} vs {baseline_precision:.4f} | "
              f"F1: {metrics['F1 Score (Float32)']:.4f} | "
              f"AUC: {metrics['ROC AUC (Float32)']:.4f} | "
              f"MCC: {metrics['MCC (Float32)']:.4f} | "
              f"Time: {elapsed_inf:.3f}s | Size: {model_size_kb:.1f} KB")

        # ROC + PR curve
        plot_prefix = plot_dir / f"top_{top_k:02d}"
        plot_precision_recall_roc(y_test, y_scores, out_path_prefix=plot_prefix)

        # Feature importance
        plot_feature_importances(
            model=model,
            X_valid=X_valid,
            y_valid=y_valid,
            feature_names=selected,
            save_path=plot_dir / f"feature_importance_top_{top_k}.png"
        )

        # Validation vs Test comparison
        val_metrics = evaluate_predictions(y_valid, model.predict(X_valid), model.predict_proba(X_valid))
        test_metrics = evaluate_predictions(y_test, y_pred, y_scores)
        plot_test_vs_validation_metrics(
            val_metrics,
            test_metrics,
            save_path=plot_dir / f"val_vs_test_top_{top_k:02d}.png"
        )

        # Log results
        results.append({
            "TopK": top_k,
            "Mode": "Float32",
            "Baseline Accuracy": baseline_accuracy,
            "Baseline Precision": baseline_precision,
            **metrics
        })

    results_df = pd.DataFrame(results)
    csv_path = out_dir / "dnn_initial_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n📊 Saved DNN initial results to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", type=str, default=str(config.Paths.features_parquet))
    parser.add_argument("--out_path", type=str, default=str(config.Paths.dnn_results))
    parser.add_argument("--topk_start", type=int, default=config.topk_start)
    parser.add_argument("--topk_end", type=int, default=config.topk_end)
    parser.add_argument("--topk_step", type=int, default=config.topk_step)
    args = parser.parse_args()

    df = get_or_generate_features(path=args.features_path)
    feature_cols = [col for col in df.columns if col.startswith("f_")]
    ranked_features = sorted(feature_cols)
    print(f"✅ Using {len(ranked_features)} features directly from feature set.")

    run_dnn_topk(df, ranked_features, Path(args.out_path), args.topk_start, args.topk_end, args.topk_step)
