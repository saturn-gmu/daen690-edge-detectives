# ptq_pipeline.py — Post-Training Quantization pipeline with INT8 evaluation + Parquet fallback

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_METAL_DISABLE"] = "1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Project root

import argparse
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import matthews_corrcoef

from src.config import config
from src.data.generate_features import get_or_generate_features
from src.models.dnn_model import DNNClassifier
from src.utils.utils import split_and_scale
from src.evaluation.evaluate_model import evaluate_predictions
from src.utils.tflite_utils import run_tflite_inference, convert_model_to_tflite, representative_dataset_gen


def main(args):
    features_path = Path(args.features_path)
    ranked_path = Path(args.ranked_path)
    out_dir = Path(args.out_path)
    rep_dir = out_dir / "rep_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # ------------------ Load Features ------------------
    if features_path.exists():
        print(f"📦 Loading cached features from: {features_path}")
        df = pd.read_parquet(features_path)
    else:
        print("🚧 Generating new features...")
        df = get_or_generate_features(
            features_path=features_path,
            ranked_path=ranked_path
        )[0]

    # ------------------ Load Ranked Features ------------------
    ranked_features = pd.read_csv(ranked_path)["feature"].tolist()
    feature_cols = [f for f in ranked_features if f in df.columns]

    results = []

    for top_k in range(args.topk_start, args.topk_end, args.topk_step):
        selected = feature_cols[:top_k]
        print(f"\n🔍 Evaluating INT8 for Top-{top_k} features...")

        X = df[selected]
        y = df["target"]
        _, splits = split_and_scale(X, y)
        X_train, X_test, X_valid = splits["X_train"], splits["X_test"], splits["X_valid"]
        y_train, y_test, y_valid = splits["y_train"], splits["y_test"], splits["y_valid"]

        # ------------------ FLOAT32 Training + Export ------------------
        K.clear_session()
        model = DNNClassifier(input_shape=(X_train.shape[1],))
        model.fit(X_train, y_train, X_valid, y_valid)

        float32_path = out_dir / f"float32_top_{top_k}.tflite"
        convert_model_to_tflite(model.model, float32_path, quantize=False)

        float32_start = time.time()
        y_pred_f32, y_scores_f32 = run_tflite_inference(float32_path, X_test)
        float32_elapsed = round(time.time() - float32_start, 4)

        f32_metrics = evaluate_predictions(y_test, y_pred_f32, y_scores_f32)
        f32_metrics = {f"{k} (Float32)": v for k, v in f32_metrics.items()}
        f32_metrics["MCC (Float32)"] = matthews_corrcoef(y_test, y_pred_f32)
        f32_metrics["Elapsed (s)"] = float32_elapsed
        f32_metrics["Model Size (KB)"] = round(float32_path.stat().st_size / 1024, 2)

        results.append({
            "TopK": top_k,
            "Mode": "Float32",
            **f32_metrics
        })

        # ------------------ Save Representative Data ------------------
        rep_data_path = rep_dir / f"rep_data_top_{top_k}.npz"
        np.savez_compressed(rep_data_path, X=X_train.astype(np.float32))

        # ------------------ INT8 Export + Inference ------------------
        int8_path = out_dir / f"int8_top_{top_k}.tflite"
        rep_data = np.load(rep_data_path)
        rep_fn = representative_dataset_gen(rep_data["X"])
        convert_model_to_tflite(model.model, int8_path, quantize=True, representative_data=rep_fn)

        int8_start = time.time()
        y_pred_int8, y_scores_int8 = run_tflite_inference(int8_path, X_test)
        int8_elapsed = round(time.time() - int8_start, 4)

        int8_metrics = evaluate_predictions(y_test, y_pred_int8, y_scores_int8)
        int8_metrics = {f"{k} (INT8)": v for k, v in int8_metrics.items()}
        int8_metrics["MCC (INT8)"] = matthews_corrcoef(y_test, y_pred_int8)
        int8_metrics["Elapsed (s)"] = int8_elapsed
        int8_metrics["Model Size (KB)"] = round(int8_path.stat().st_size / 1024, 2)
        int8_metrics["Speedup (x)"] = round(float32_elapsed / int8_elapsed, 2) if int8_elapsed else None

        results.append({
            "TopK": top_k,
            "Mode": "INT8",
            **int8_metrics
        })

        print(f"✅ Top-{top_k} done | Float32: {float32_elapsed}s | INT8: {int8_elapsed}s | Size: {int8_metrics['Model Size (KB)']} KB")

    # ------------------ Save Results ------------------
    try:
        results_df = pd.DataFrame(results)
        csv_path = out_dir / "ptq_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n📊 Saved PTQ results to: {csv_path}")
    except Exception as e:
        print(f"\n❌ Failed to save CSV: {e}")

    print("✅ All evaluations complete.")


# ------------------ CLI Entry ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", type=str, default=str(config.Paths.features_parquet))
    parser.add_argument("--ranked_path", type=str, default=str(config.Paths.ranked_features_csv))
    parser.add_argument("--out_path", type=str, default=str(config.Paths.ptq_results))
    parser.add_argument("--topk_start", type=int, default=config.TopK.start)
    parser.add_argument("--topk_end", type=int, default=config.TopK.end)
    parser.add_argument("--topk_step", type=int, default=config.TopK.step)
    args = parser.parse_args()
    main(args)
