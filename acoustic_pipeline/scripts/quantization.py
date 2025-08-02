# === scripts/quantization.py ===
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # Project root

from src.config import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ------------------ Representative Dataset Generator ------------------
def representative_dataset_gen(rep_data_path):
    data = np.load(rep_data_path)
    X = data['X'] if 'X' in data else data[list(data.keys())[0]]

    def generator():
        for i in range(min(100, len(X))):
            yield [np.expand_dims(X[i].astype(np.float32), axis=0)]
    return generator

# ------------------ INT8 Quantization ------------------
def quantize_h5_model(h5_path, output_path, rep_data_path):
    print(f"🔧 Converting {h5_path.name} → {output_path.name}")
    model = tf.keras.models.load_model(h5_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(rep_data_path)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ Saved INT8 model to {output_path}\n")

# ------------------ CLI Entry ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch quantize .h5 models to INT8 .tflite")
    parser.add_argument("--input_dir", type=str, default=str(config.Paths.h5_models_dir))
    parser.add_argument("--output_dir", type=str, default=str(config.Paths.ptq_models_dir))
    parser.add_argument("--rep_data_subdir", type=str, default="rep_data")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    rep_dir = in_dir / args.rep_data_subdir

    out_dir.mkdir(parents=True, exist_ok=True)

    h5_models = sorted(in_dir.glob("dnn_top_*.h5"))
    if not h5_models:
        raise FileNotFoundError(f"No .h5 models found in {in_dir}")

    for h5_path in h5_models:
        top_k = h5_path.stem.split("_")[-1]
        rep_data_path = rep_dir / f"rep_data_top_{top_k}.npz"
        if not rep_data_path.exists():
            print(f"⚠️ Skipping {h5_path.name}: Missing {rep_data_path.name}")
            continue

        out_path = out_dir / f"int8_top_{top_k}.tflite"
        quantize_h5_model(h5_path, out_path, rep_data_path)
