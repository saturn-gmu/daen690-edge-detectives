# plot_ptq_summary.py — Visualize PTQ + DNN results and generate plots

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.config.config import Paths


def plot_ptq_summary():
    ptq_path = Paths.ptq_results / "ptq_results.csv"
    dnn_path = Paths.dnn_results / "pipeline_metrics.csv"
    out_path = Paths.ptq_plot_dir
    merged_csv_path = out_path.parent / "ptq_plus_dnn.csv"

    out_path.mkdir(parents=True, exist_ok=True)

    if not ptq_path.exists():
        raise FileNotFoundError(f"❌ PTQ results CSV not found at {ptq_path}")

    df = pd.read_csv(ptq_path)

    if dnn_path.exists():
        dnn_df = pd.read_csv(dnn_path)
        if "TopK" not in dnn_df.columns:
            dnn_df["TopK"] = [40] * len(dnn_df)
        dnn_df["Mode"] = "Float32"
        ordered_cols = ["TopK", "Mode", "Elapsed (s)", "Model Size (KB)"]
        ordered_cols += [col for col in dnn_df.columns if col not in ordered_cols]
        dnn_df = dnn_df[ordered_cols]
        existing_topks = df[df["Mode"] == "Float32"]["TopK"].unique()
        dnn_df = dnn_df[~dnn_df["TopK"].isin(existing_topks)]
        if not dnn_df.empty:
            print(f"📥 Merging {len(dnn_df)} Float32 rows from DNN Initial...")
            df = pd.concat([df, dnn_df], ignore_index=True)

    df.to_csv(merged_csv_path, index=False)
    print(f"✅ Merged PTQ + DNN CSV saved to: {merged_csv_path}")

    # Generate plots
    plot_all_metrics_combined(df, out_path / "ptq_combined_metrics.png")
    plot_inference_time_comparison(df, out_path / "inference_time_comparison.png")
    plot_speedup_vs_topk(df, out_path / "speedup_vs_topk.png")
    plot_model_size_comparison(df, out_path / "model_size_vs_topk.png")
    plot_each_metric_by_mode(df, out_path)
    plot_file_size_comparison(df, out_path / "file_size_comparison.png")
    plot_int8_inference_time_only(df, out_path / "int8_inference_time_vs_topk.png")


def plot_all_metrics_combined(df, out_path):
    metric_cols = [col for col in df.columns if col in ["Accuracy", "Precision", "AUC", "F1 Score"]]
    melted = pd.melt(df, id_vars=["TopK"], value_vars=metric_cols, var_name="Metric", value_name="Value")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=melted, x="TopK", y="Value", hue="Metric", marker="o")
    plt.title("Metrics vs Top-K Features")
    plt.xlabel("Top-K Features")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Combined metrics plot saved to: {out_path}")


def plot_each_metric_by_mode(df, out_path):
    if "TopK" not in df.columns:
        print("⚠️ Missing TopK column. Skipping metric plots.")
        return

    metric_cols = [col for col in df.columns if "(" in col]
    melted = pd.melt(df, id_vars=["TopK"], value_vars=metric_cols, var_name="Metric", value_name="Value")

    for metric in melted["Metric"].unique():
        data_subset = melted[melted["Metric"] == metric]
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=data_subset, x="TopK", y="Value", marker="o", label=metric)
        plt.title(f"{metric} vs Top-K Features")
        plt.xlabel("Top-K Features")
        plt.ylabel(metric)
        plt.grid(True)
        plt.legend(title="Metric")
        plt.tight_layout()
        filename = f"{metric.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}_vs_topk.png"
        plt.savefig(out_path / filename)
        plt.close()
        print(f"📊 Saved: {filename}")


def plot_inference_time_comparison(df, out_path):
    if "Elapsed (s)" not in df.columns or "Mode" not in df.columns:
        print("⚠️ Skipping inference time plot: 'Elapsed (s)' or 'Mode' column missing.")
        return
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="TopK", y="Elapsed (s)", hue="Mode", marker="o")
    plt.title("Inference Time vs Top-K Features")
    plt.xlabel("Top-K Features")
    plt.ylabel("Elapsed Time (s)")
    plt.grid(True)
    plt.legend(title="Mode")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Inference time plot saved to: {out_path}")


def plot_speedup_vs_topk(df, out_path):
    speedup_df = df[df["Mode"] == "INT8"].copy()
    if "Speedup (x)" not in speedup_df.columns:
        print("⚠️ 'Speedup (x)' column missing, skipping speedup plot.")
        return
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=speedup_df, x="TopK", y="Speedup (x)", marker="o")
    plt.title("INT8 Inference Speedup vs Top-K")
    plt.xlabel("Top-K Features")
    plt.ylabel("Speedup (Float32 / INT8)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Speedup plot saved to: {out_path}")


def plot_model_size_comparison(df, out_path):
    if "Model Size (KB)" not in df.columns or "Mode" not in df.columns:
        print("⚠️ 'Model Size (KB)' or 'Mode' column missing, skipping model size plot.")
        return
    size_df = df[["TopK", "Mode", "Model Size (KB)"]].dropna().copy()
    size_df.loc[size_df["Mode"] == "Float32", "Model Size (KB)"] += 0.1  # Offset for visibility

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=size_df,
        x="TopK",
        y="Model Size (KB)",
        hue="Mode",
        style="Mode",
        markers=True,
        dashes={"Float32": (2, 2), "INT8": ""}
    )
    plt.title("Model Size vs Top-K Features")
    plt.xlabel("Top-K Features")
    plt.ylabel("Model Size (KB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Model size plot saved to: {out_path}")


def plot_file_size_comparison(df, out_path):
    if "Model Size (KB)" not in df.columns or "Mode" not in df.columns:
        print("⚠️ Skipping file size comparison: required columns missing.")
        return
    size_df = df[["TopK", "Mode", "Model Size (KB)"]].copy()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=size_df, x="TopK", y="Model Size (KB)", hue="Mode")
    plt.title("Model File Size Comparison: Float32 vs INT8")
    plt.xlabel("Top-K Features")
    plt.ylabel("Size (KB)")
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig(out_path)
    plt.close()
    print(f"📊 File size comparison plot saved to: {out_path}")


def plot_int8_inference_time_only(df, out_path):
    int8_df = df[df["Mode"] == "INT8"]
    if int8_df.empty:
        print("⚠️ No INT8 data found, skipping INT8-only inference time plot.")
        return
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=int8_df, x="TopK", y="Elapsed (s)", marker="o", color="orange")
    plt.title("INT8 Inference Time vs Top-K Features")
    plt.xlabel("Top-K Features")
    plt.ylabel("Elapsed Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"📊 INT8-only inference time plot saved to: {out_path}")


# ------------------ ENTRY POINT ------------------
if __name__ == "__main__":
    plot_ptq_summary()
