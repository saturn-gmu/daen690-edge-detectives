# pipeline_quant_top20.py
# Trains a DNN on top-20 bottleneck features and quantizes it to INT8

from src.dataloader import load_data_and_extract_features
from src.dnn_model import train_dnn_model, extract_bottleneck_features
from src.utils import start_timer, end_timer
from src.evaluation import evaluate_model, evaluate_tflite_model
from src.config import folder_path, vesselnames, vessel

from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import time
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

# --- Start timing ---
total_start = start_timer()
print("🚀 Training and quantizing DNN using top-20 bottleneck features")

# --- Load top 20 mask and dataset ---
data_start = time.time()
feature_mask = np.load("artifacts/thresholded/feature_mask_20.npy")
df = load_data_and_extract_features(folder_path, vesselnames, vessel)
X_full = np.vstack(df['features'].values)
y = df['target'].values

# --- Extract 64D bottleneck features ---
model_full = load_model("artifacts/quant_dnn/dnn_model.h5")
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X_full)
bottleneck_model = load_model("artifacts/quant_dnn/dnn_model.h5")
X_train_b, X_test_b, X_all_b = extract_bottleneck_features(bottleneck_model, X_full_scaled, X_full_scaled, scaler_full, df)
X = X_all_b[:, feature_mask]
data_end = time.time()

# --- Split + scale ---
split_start = time.time()
X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.4, random_state=42)
X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_valid_scaled = scaler.transform(X_valid)
split_end = time.time()

os.makedirs("artifacts/quant_top_features", exist_ok=True)
os.makedirs("artifacts/tflite", exist_ok=True)
np.savez("artifacts/quant_top_features/features_scaled.npz", X_train=X_train_scaled, X_test=X_test_scaled,
         X_valid=X_valid_scaled, y_train=y_train, y_test=y_test, y_valid=y_valid)
joblib.dump(scaler, "artifacts/quant_top_features/feature_scaler.pkl")

# --- Train DNN ---
train_start = time.time()
model, history = train_dnn_model(X_train_scaled, y_train, X_valid_scaled, y_valid)
model.save("artifacts/quant_top_features/dnn_model.h5")
train_end = time.time()

# --- Evaluate original model ---
eval_start = time.time()
y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
print(f"✅ Accuracy (float32): {acc:.4f}, Precision: {prec:.4f}")
eval_end = time.time()

# --- Quantize model to INT8 ---
quant_start = time.time()
def representative_data_gen():
    for i in range(min(100, len(X_train_scaled))):
        yield [X_train_scaled[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

quant_model = converter.convert()
with open("artifacts/tflite/dnn_model_top20_int8.tflite", "wb") as f:
    f.write(quant_model)
print("✅ Quantized model saved to artifacts/tflite/dnn_model_top20_int8.tflite")
quant_end = time.time()

# --- Evaluate INT8 model ---
tflite_eval_start = time.time()
evaluate_tflite_model("artifacts/tflite/dnn_model_top20_int8.tflite", X_test_scaled, y_test)
tflite_eval_end = time.time()

# --- Timing Summary ---
print(f"⏱️ Data loading time: {data_end - data_start:.2f} seconds")
print(f"⏱️ Split & scale time: {split_end - split_start:.2f} seconds")
print(f"⏱️ Training time: {train_end - train_start:.2f} seconds")
print(f"⏱️ Evaluation time (float32): {eval_end - eval_start:.2f} seconds")
print(f"⏱️ Quantization time: {quant_end - quant_start:.2f} seconds")
print(f"⏱️ Evaluation time (INT8): {tflite_eval_end - tflite_eval_start:.2f} seconds")

# --- End timing ---
end_timer(total_start)
print("✅ Full training + quantization pipeline complete")
