# src/utils/tflite_utils.py

import numpy as np
import tensorflow as tf

def run_tflite_inference(model_path, X_scaled, verbose=False):
    """
    Runs inference on a TFLite model with either float32 or int8 input.
    Returns predicted labels and raw model outputs.
    """
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_dtype = input_details["dtype"]
    input_index = input_details["index"]

    if input_dtype == np.int8:
        scale, zp = input_details["quantization"]
        if scale == 0:
            raise ValueError("🚫 Invalid quantization scale (0). Cannot quantize input.")
        X_processed = ((X_scaled / scale) + zp).astype(np.int8)
    else:
        X_processed = X_scaled.astype(np.float32)

    y_scores, y_preds = [], []
    for i, x in enumerate(X_processed):
        interpreter.set_tensor(input_index, np.expand_dims(x, axis=0))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0][0]
        y_scores.append(output)
        y_preds.append(int(output > 0.5))
        if verbose and i < 5:
            print(f"Sample {i} → score: {output:.4f}, pred: {int(output > 0.5)}")

    return np.array(y_preds), np.array(y_scores)


def convert_model_to_tflite(model, output_path, quantize=True, representative_data=None):
    """
    Converts a Keras model to TFLite.
    Supports optional full INT8 quantization using a representative dataset.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_data is not None:
            converter.representative_dataset = representative_data
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        else:
            print("⚠️ Warning: Quantization enabled, but no representative dataset provided. Using dynamic range.")

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"💾 Saved TFLite model to {output_path}")


def representative_dataset_gen(X_sample):
    """
    Creates a generator from a NumPy array for use as a representative dataset.
    """
    def gen():
        for i in range(min(len(X_sample), 100)):
            yield [np.expand_dims(X_sample[i].astype(np.float32), axis=0)]
    return gen


def representative_dataset_from_npz(npz_path):
    """
    Loads a representative dataset generator from a .npz file with array 'X'.
    """
    data = np.load(npz_path)
    X = data["X"] if "X" in data else data[list(data.keys())[0]]

    return representative_dataset_gen(X)
