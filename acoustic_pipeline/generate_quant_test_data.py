# generate_quantized_test_data.py
# Quantizes X_test to INT8 using model input details and saves .npz for future use

import numpy as np
import tensorflow as tf

# Load test data (float32 scaled features)
data = np.load("artifacts/quant_top_features/features_scaled.npz")
X_test = data['X_test']
y_test = data['y_test']

# Load TFLite model to get input quantization parameters
interpreter = tf.lite.Interpreter(model_path="artifacts/tflite/dnn_model_top20_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()

scale, zero_point = input_details[0]['quantization']

# Quantize X_test to INT8
X_test_int8 = (X_test / scale + zero_point).astype(np.int8)

# Save to .npz for quick loading during comparison
np.savez("artifacts/tflite/quantized_test_data.npz", X_test_int8=X_test_int8, y_test=y_test)
print("✅ Saved: artifacts/tflite/quantized_test_data.npz")