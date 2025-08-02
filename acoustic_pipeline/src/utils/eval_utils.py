import numpy as np
import tensorflow as tf
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
from pathlib import Path


def evaluate_model(model, X, y):
    y_pred_probs = model.predict(X)
    y_pred = np.argmax(y_pred_probs, axis=1)
    return {
        "accuracy": accuracy_score(y, y_pred)
    }


def convert_to_tflite(keras_model_path, output_path, quantize=False):
    model = tf.keras.models.load_model(keras_model_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)


def measure_tflite_inference(tflite_path, X, y):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []
    start = time.time()

    for sample in X:
        input_data = np.expand_dims(sample, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        pred = np.argmax(output)
        predictions.append(pred)

    end = time.time()
    acc = accuracy_score(y, predictions)
    latency = (end - start) / len(X)

    return {
        "accuracy": acc,
        "latency_per_sample": latency
    }


def evaluate_predictions(y_true, y_pred, y_scores=None, include_mcc=True, verbose=False):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_scores is not None:
        metrics["ROC AUC"] = roc_auc_score(y_true, y_scores)
        metrics["PR AUC"] = average_precision_score(y_true, y_scores)
    if include_mcc:
        metrics["MCC"] = matthews_corrcoef(y_true, y_pred)
    if verbose:
        print("📊 Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    return metrics


def plot_precision_recall_roc(y_true, y_scores, out_path_prefix="results/plot"):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    roc_path = f"{out_path_prefix}_roc.png"
    plt.savefig(roc_path)
    plt.close()
    print(f"📈 Saved ROC curve to {roc_path}")

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    pr_path = f"{out_path_prefix}_pr.png"
    plt.savefig(pr_path)
    plt.close()
    print(f"📉 Saved Precision-Recall curve to {pr_path}")


def plot_training_history(history, save_path="training_history.png"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_metric_over_epochs(metrics_by_epoch, save_path, title, ylabel):
    plt.figure()
    for label, values in metrics_by_epoch.items():
        plt.plot(values, label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
