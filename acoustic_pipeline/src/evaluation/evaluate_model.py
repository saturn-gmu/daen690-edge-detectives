import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

# Import essential classification evaluation metrics from sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve,
    auc
)

def evaluate_predictions(y_true, y_pred, y_scores=None, include_mcc=True, verbose=False):
    """
    Evaluate classification results using common metrics.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.
        y_scores (array-like, optional): Probability scores for the positive class.
        include_mcc (bool): Whether to include Matthews Correlation Coefficient.
        verbose (bool): If True, print metrics to console.

    Returns:
        dict: Dictionary containing evaluation metric names and values.
    """
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
    """
    Generate and save ROC and Precision-Recall curves for classification performance.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_scores (array-like): Predicted probability scores.
        out_path_prefix (str): Base path (no extension) for saving the plots.
    """
    # ROC Curve
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

    # Precision-Recall Curve
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
    """
    Plots training and validation accuracy/loss over epochs.

    Args:
        history: Keras History object returned by model.fit()
        save_path (str or Path): File path to save the plot
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    # Loss plot
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
    base_scores = model.predict_proba(X_valid)
    base_auc = roc_auc_score(y_valid, base_scores)

    importances = []
    for i in range(X_valid.shape[1]):
        X_permuted = X_valid.copy()
        np.random.shuffle(X_permuted[:, i])
        permuted_scores = model.predict_proba(X_permuted)
        score = roc_auc_score(y_valid, permuted_scores)
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

def plot_test_vs_validation_metrics(val_metrics, test_metrics, save_path="results/accuracy_precision_compare.png"):
    """
    Plots a bar chart comparing Accuracy and Precision on validation and test sets.

    Args:
        val_metrics (dict): Metrics dictionary from validation set (must include 'Accuracy', 'Precision').
        test_metrics (dict): Metrics dictionary from test set (must include 'Accuracy', 'Precision').
        save_path (str or Path): Path to save the plot.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ['Accuracy', 'Precision']
    val_values = [val_metrics.get('Accuracy', 0), val_metrics.get('Precision', 0)]
    test_values = [test_metrics.get('Accuracy', 0), test_metrics.get('Precision', 0)]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 5))
    plt.bar(x - width/2, val_values, width, label='Validation', color='steelblue')
    plt.bar(x + width/2, test_values, width, label='Test', color='darkorange')

    plt.ylabel('Score')
    plt.title('Accuracy and Precision: Validation vs Test')
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Saved validation vs test metric comparison plot to {save_path}")
