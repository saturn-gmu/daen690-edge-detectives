import tensorflow as tf
    
def convert_model_to_tflite(model, output_path, quantize=False, representative_data=None):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    """
    Converts a Keras model to TFLite format with optional post-training quantization.

    Args:
        model (tf.keras.Model): Trained Keras model.
        output_path (str or Path): Path to save the converted TFLite model.
        quantize (bool): If True, applies INT8 quantization.
        representative_data (callable): Generator for representative dataset for INT8 calibration.
    """
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if representative_data is not None:
            converter.representative_dataset = representative_data
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        else:
            print("⚠️ No representative dataset provided. INT8 quantization may fall back to dynamic range.")

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ TFLite model saved to {output_path}")