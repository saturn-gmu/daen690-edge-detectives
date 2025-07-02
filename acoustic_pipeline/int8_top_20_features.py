# inference_top20.py
# Lightweight script to run inference on TFLite INT8 top-20 bottleneck DNN model

import tensorflow as tf
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# --- Load INT8 model ---
model_path = "artifacts/tflite/dnn_model_top20_int8.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Load scaler and test data ---
data = np.load("artifacts/quant_top_features/features_scaled.npz")
X_test = data["X_test"]
y_test = data["y_test"]

# --- Quantization parameters ---
input_scale, input_zero_point = input_details[0]['quantization']

# --- Run inference ---
y_pred = []
for x in X_test:
    x_int8 = (x / input_scale + input_zero_point).astype(np.int8)
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(x_int8, axis=0))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    y_pred.append(np.argmax(output))

# --- Output predictions ---
print("✅ Predictions complete")
print("First 10 predictions:", y_pred[:10])
print("True labels:", y_test[:10])