import sys
from pathlib import Path
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tqdm import tqdm
import csv
import time
import gc

# Disable GPU to avoid CUDA registration conflicts
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Add project root to PYTHONPATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.dnn_model import DNNClassifier
from src.utils.data_utils import load_feature_dataframe, get_top_k_features
from src.utils.eval_utils import evaluate_model, convert_to_tflite, measure_tflite_inference

def run_prune_and_quantize(df, ranked_features, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summary_csv = os.path.join(out_dir, "ptq_summary.csv")

    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Top_K", "Float_Acc", "INT8_Acc", "Float_Size_KB", "INT8_Size_KB", "Train_Time_Sec"])

        top_k_list = [5]  # Limit to Top-5 for debugging stability

        for k in tqdm(top_k_list, desc="🔄 Sweeping Top-K Features"):
            try:
                top_k_cols = get_top_k_features(ranked_features, k)
                X = df[top_k_cols].values.astype(np.float32)
                y = df['label'].values.astype(np.int64)
                num_classes = len(np.unique(y))

                from sklearn.model_selection import train_test_split
                X_train, X_valid, y_train, y_valid = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                pruning_params = {
                    "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                        initial_sparsity=0.2,
                        final_sparsity=0.8,
                        begin_step=0,
                        end_step=1000,
                    )
                }

                print(f"\n✨ Top-{k}: Training pruned DNN...")
                model = DNNClassifier(
                    input_shape=X_train.shape[1:],
                    num_classes=num_classes,
                    prune=True,
                    pruning_params=pruning_params,
                )

                pruning_cb = tfmot.sparsity.keras.UpdatePruningStep()
                early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

                start_time = time.time()
                model.fit(X_train, y_train, X_valid, y_valid,
                          callbacks=[pruning_cb, early_stop],
                          epochs=5, batch_size=8, verbose=1)
                elapsed = time.time() - start_time

                float_path = os.path.join(out_dir, f"float_top{k}.h5")
                model.save(float_path)

                float_metrics = evaluate_model(model.model, X_valid, y_valid)

                int8_path = os.path.join(out_dir, f"int8_top{k}.tflite")
                convert_to_tflite(float_path, int8_path, quantize=True)
                int8_metrics = measure_tflite_inference(int8_path, X_valid, y_valid)

                float_size = os.path.getsize(float_path) / 1024
                int8_size = os.path.getsize(int8_path) / 1024

                print(f"✅ Top-{k} | ACC Float: {float_metrics['accuracy']:.3f} | INT8: {int8_metrics['accuracy']:.3f}")
                print(f"📦 Sizes (KB): Float={float_size:.1f}, INT8={int8_size:.1f} | ⏱️ Time: {elapsed:.1f}s")

                writer.writerow([k, float_metrics['accuracy'], int8_metrics['accuracy'],
                                 float_size, int8_size, round(elapsed, 2)])

                # Aggressively clean up
                del model
                tf.keras.backend.clear_session()
                gc.collect()
                time.sleep(2)

            except Exception as e:
                print(f"❌ Skipping Top-{k} due to error: {e}")
                writer.writerow([k, "ERROR", "ERROR", "-", "-", "-"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save output models")
    parser.add_argument("--features_path", type=str, default="artifacts/features_df.parquet")
    parser.add_argument("--ranked_path", type=str, default="artifacts/ablation/ranked_features.csv")
    args = parser.parse_args()

    df = load_feature_dataframe(path=args.features_path)
    if not os.path.exists(args.ranked_path):
        raise FileNotFoundError(f"Ranked features file not found: {args.ranked_path}")
    ranked_features = pd.read_csv(args.ranked_path)

    run_prune_and_quantize(df, ranked_features, args.out_dir)

if __name__ == "__main__":
    main()
