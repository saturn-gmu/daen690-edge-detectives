import pandas as pd
import matplotlib.pyplot as plt
import os

# === Configurable highlight parameters ===
highlight_range = (18, 22)   # Example: range of most important features
annotate_k = 20              # Optional: highlight Top-20

# === Load metrics ===
metrics_path = "results/pipeline_metrics.csv"
if not os.path.exists(metrics_path):
    raise FileNotFoundError(f"❌ Could not find {metrics_path}")

df = pd.read_csv(metrics_path)
df["Top-K"] = pd.to_numeric(df["Model"].str.extract(r"Top-(\d+)")[0], errors="coerce")
df = df.dropna(subset=["Top-K"]).sort_values("Top-K")

# === Plot 1: Accuracy and Precision ===
plt.figure(figsize=(10, 6))
plt.plot(df["Top-K"], df["Accuracy"], marker='o', label="Accuracy")
plt.plot(df["Top-K"], df["Precision"], marker='s', label="Precision")
plt.axvspan(highlight_range[0], highlight_range[1], color='lightgreen', alpha=0.3, label="Important Feature Range")

if annotate_k in df["Top-K"].values:
    y_val = df.loc[df["Top-K"] == annotate_k, "Accuracy"].values[0]
    plt.annotate(f"Top-{annotate_k}", xy=(annotate_k, y_val),
                 xytext=(annotate_k+1, y_val+0.01),
                 arrowprops=dict(arrowstyle="->", color='black'),
                 fontsize=10, color='darkgreen')

plt.xlabel("Number of Features (Top-K)")
plt.ylabel("Score")
plt.title("Accuracy & Precision vs Number of Features")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/ptq_accuracy_precision_plot.png")
print("📈 Saved plot to results/ptq_accuracy_precision_plot.png")
plt.close()

# === Plot 2: Inference Time ===
plt.figure(figsize=(10, 6))
plt.plot(df["Top-K"], df["Inference Time (ms)"], marker='^', color="darkorange", label="INT8 Inference Time")
plt.axvspan(highlight_range[0], highlight_range[1], color='lightgreen', alpha=0.3, label="Important Feature Range")

if annotate_k in df["Top-K"].values:
    y_val = df.loc[df["Top-K"] == annotate_k, "Inference Time (ms)"].values[0]
    plt.annotate(f"Top-{annotate_k}", xy=(annotate_k, y_val),
                 xytext=(annotate_k+2, y_val+1),
                 arrowprops=dict(arrowstyle="->", color='black'),
                 fontsize=10, color='darkgreen')

plt.xlabel("Number of Features (Top-K)")
plt.ylabel("Inference Time (ms per sample)")
plt.title("INT8 Inference Time vs Number of Features")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/ptq_inference_time_plot.png")
print("📈 Saved plot to results/ptq_inference_time_plot.png")
plt.close()

