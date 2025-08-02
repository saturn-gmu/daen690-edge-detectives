from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from pathlib import Path

def split_and_scale(X, y, use_minmax=False):
    """
    Splits data into train, test, and validation sets, then scales features.

    Parameters:
    - X (pd.DataFrame or np.ndarray): Input features
    - y (pd.Series or np.ndarray): Target labels
    - use_minmax (bool): Whether to use MinMaxScaler instead of StandardScaler

    Returns:
    - scaler (fitted Scaler): The fitted scaler object
    - dict: Dictionary containing split and scaled data
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=0.4, random_state=42)
    X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)
    scaler = MinMaxScaler() if use_minmax else StandardScaler()
    return scaler, {
        "X_train": scaler.fit_transform(X_train),
        "X_test": scaler.transform(X_test),
        "X_valid": scaler.transform(X_valid),
        "y_train": y_train,
        "y_test": y_test,
        "y_valid": y_valid
    }

def save_metrics_csv(filepath, k, dnn_metrics=None, tflite_metrics=None, float_metrics=None):
    """
    Appends evaluation metrics to a CSV file.

    Parameters:
    - filepath (Path or str): Path to the CSV file
    - k (int): Number of top features used
    - dnn_metrics (dict, optional): DNN evaluation results
    - tflite_metrics (dict, optional): TFLite INT8 evaluation results
    - float_metrics (dict, optional): TFLite Float32 evaluation results
    """
    filepath = Path(filepath)
    results = {"TopK": k}

    if dnn_metrics:
        results.update({f"DNN_{key}": val for key, val in dnn_metrics.items()})
    if tflite_metrics:
        results.update({f"TFLite_INT8_{key}": val for key, val in tflite_metrics.items()})
    if float_metrics:
        results.update({f"TFLite_F32_{key}": val for key, val in float_metrics.items()})

    df = pd.DataFrame([results])

    if filepath.exists():
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, index=False)

def log_metrics(name, metrics):
    print(f"\n📊 {name} Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

def extract_label_from_filename(filename):
    # Example: "_No7_" becomes class "No7", fallback = "Unknown"
    parts = filename.split("_")
    for part in parts:
        if "No" in part:
            return part
    return "Unknown"
