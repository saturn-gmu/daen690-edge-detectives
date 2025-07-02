# Centralized evaluation utilities: Keras, TFLite, history, and feature plots

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score
import tensorflow as tf

def evaluate_model(model, X_test, y_test, title="Model", vessel=None):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    print(f"\n{title} Evaluation:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {title}")
    plt.tight_layout()
    plt.show()

def evaluate_tflite_model(tflite_path, X_test, y_test):
    print(f"Evaluating TFLite model: {tflite_path}")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    scale, zero_point = input_details[0]['quantization']
    X_test_q = (X_test / scale + zero_point).astype(np.int8)

    y_pred = []
    for x in X_test_q:
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(x, axis=0))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        y_pred.append(np.argmax(output))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    print(f"✅ TFLite Model Accuracy: {acc:.4f}, Precision: {prec:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Oranges)
    plt.title("Confusion Matrix: TFLite INT8 Model")
    plt.tight_layout()
    plt.show()

    return acc, prec


def plot_bottleneck_feature_importance(importances, top_n=20):
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 4))
    plt.bar(range(top_n), importances[sorted_idx])
    plt.xticks(range(top_n), sorted_idx, rotation=45)
    plt.title("Top Bottleneck Feature Importances")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()



def plot_history(history):
    # Accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Loss
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Precision
    if 'precision' in history.history:
        plt.figure(figsize=(10, 4))
        plt.plot(history.history['precision'], label='Train Precision')
        if 'val_precision' in history.history:
            plt.plot(history.history['val_precision'], label='Val Precision')
        plt.title("Precision Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Precision")
        plt.legend()
        plt.tight_layout()
        plt.show()


    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Optional: Compare accuracy between train/test sets
def compare_accuracy(model, X_train, X_test, y_train, y_test):
    y_pred_train = np.argmax(model.predict(X_train), axis=1)
    y_pred_test = np.argmax(model.predict(X_test), axis=1)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)

    plt.figure(figsize=(5, 4))
    plt.bar(["Train", "Test"], [acc_train, acc_test], color=['steelblue', 'salmon'])
    plt.ylim(0.0, 1.0)
    plt.title("Train vs Test Accuracy")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()

    print(f"✅ Train Accuracy: {acc_train:.4f}, Test Accuracy: {acc_test:.4f}")

# RF plot utility (must pass trained RF model)
from src.rf_model import plot_cumulative_and_dynamic_importance

def plot_rf_importance_summary(rf_model):
    plot_cumulative_and_dynamic_importance(rf_model)