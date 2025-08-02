# dnn_evaluation.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score

def plot_training_history(history, save_path="training_history.png"):
    """
    Plots training and validation accuracy/loss over epochs.

    Args:
        history: Keras History object returned by model.fit()
        save_path (str or Path): File path to save the plot
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))

    # --- Accuracy plot ---
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    # --- Loss plot ---
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

def plot_feature_importances(model, X_valid, y_valid, feature_names, save_path):
    """
    Estimates feature importance for a Keras model using permutation importance on validation data.

    Args:
        model (tf.keras.Model): Trained Keras model
        X_valid (pd.DataFrame or np.ndarray): Validation feature data
        y_valid (np.ndarray): Ground truth labels
        feature_names (list): List of feature names
        save_path (str or Path): Where to save the plot
    """
    from copy import deepcopy

    base_scores = model.predict_proba(X_valid)
    base_auc = roc_auc_score(y_valid, base_scores)

    importances = []
    for i in range(X_valid.shape[1]):
        X_permuted = deepcopy(X_valid)
        np.random.shuffle(X_permuted[:, i])
        score = roc_auc_score(y_valid, model.predict_proba(X_permuted))
        importances.append(base_auc - score)

    importances = np.array(importances)
    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances[sorted_idx])
    plt.xticks(range(len(importances)), np.array(feature_names)[sorted_idx], rotation=90)
    plt.ylabel("Permutation Importance (AUC drop)")
    plt.title("Estimated Feature Importance (DNN)")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
