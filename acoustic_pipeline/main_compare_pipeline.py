# main_compare_pipelines.py
# Compare performance and timing of full DNN vs quantized top-20 feature model

import numpy as np
import joblib
import time
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score

# --- Load full model and data ---
print("\n🔍 Loading full-resolution DNN model")
model_full = load_model("artifacts/quant_dnn/dnn_model.h5")
data_full = np.load("artifacts/quant_dnn/features_scaled.npz")
X_test_full = data_full['X_test']
y_test_full = data_full['y_test']

# --- Load INT8 model and INT8 quantized test data ---
print("\n🔍 Loading INT8 quantized model and pre-quantized test data")
interpreter = tf.lite.Interpreter(model_path="artifacts/tflite/dnn_model_top20_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load pre-quantized test data from saved artifact
quant_data = np.load("artifacts/tflite/quantized_test_data.npz")
X_test_int8 = quant_data['X_test_int8']
y_test_int8 = quant_data['y_test']

# --- Evaluate full model ---
print("\n⚙️ Evaluating Full Model...")
t0 = time.time()
y_pred_full = np.argmax(model_full.predict(X_test_full), axis=1)
t1 = time.time()
acc_full = accuracy_score(y_test_full, y_pred_full)
prec_full = precision_score(y_test_full, y_pred_full)
time_full = t1 - t0

# --- Evaluate INT8 model using pre-quantized input ---
print("\n⚙️ Evaluating INT8 Model...")
t2 = time.time()
y_pred_int8 = []
for x_int8 in X_test_int8:
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(x_int8, axis=0))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    y_pred_int8.append(np.argmax(output))
t3 = time.time()
acc_int8 = accuracy_score(y_test_int8, y_pred_int8)
prec_int8 = precision_score(y_test_int8, y_pred_int8)
time_int8 = t3 - t2

# --- Summary ---
print("\n📊 Performance Summary:")
print(f"Full Model    → Time: {time_full:.3f}s | Accuracy: {acc_full:.4f} | Precision: {prec_full:.4f}")
print(f"INT8 Model    → Time: {time_int8:.3f}s | Accuracy: {acc_int8:.4f} | Precision: {prec_int8:.4f}")

# --- Visualization ---
labels = ["Full Model", "INT8 Model"]
acc = [acc_full, acc_int8]
prec = [prec_full, prec_int8]

y_pos = np.arange(len(labels))

plt.figure(figsize=(10, 4))
plt.barh(y_pos - 0.15, acc, height=0.3, label="Accuracy", color="skyblue")
plt.barh(y_pos + 0.15, prec, height=0.3, label="Precision", color="lightgreen")
plt.yticks(y_pos, labels)
plt.xlabel("Score")
plt.title("Full vs INT8 Model Performance")
plt.legend()
plt.tight_layout()
plt.savefig("results/full_vs_int8_comparison.png")
plt.show()

# --- Timing Summary ---
print(f"\n⏱️ Full Model Inference Time: {time_full:.3f} seconds")
print(f"⏱️ INT8 Model Inference Time: {time_int8:.3f} seconds")