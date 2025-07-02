# rf_model.py
# Train Random Forest and visualize feature importances across multiple dimensions

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

# Train a Random Forest classifier with class balancing and 100 trees
def train_random_forest(X, y):
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X, y)
    return rf

# Plot feature importance grouped by flat frequency bins (e.g., every 10 Hz)
def plot_importance_binned_by_frequency(importances, freq_labels, bin_width=10, title="Feature Importance by Frequency Band", filename="results/frequency_band_importance.png"):
    freqs = [int(label.split()[0]) for label in freq_labels]
    binned = defaultdict(list)

    for freq, imp in zip(freqs, importances):
        lower = ((freq - 1) // bin_width) * bin_width + 1
        upper = lower + bin_width - 1
        bin_label = f"{lower}-{upper} Hz"
        binned[bin_label].append(imp)

    sorted_bins = sorted(binned.keys(), key=lambda x: int(x.split('-')[0]))
    values = [np.mean(binned[bin_label]) for bin_label in sorted_bins]

    plt.figure(figsize=(14, 6))
    plt.bar(sorted_bins, values, color="seagreen")
    plt.xlabel("Frequency Range (Hz)")
    plt.ylabel("Average Importance")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Saved: {filename}")

# Visualize RF feature importance (top-N, sorted by frequency)
def plot_rf_feature_importance(rf, top_n=400):
    n_bins = rf.feature_importances_.shape[0] // 2
    bins_per_octave = 12
    fmin = 8.0
    freqs = fmin * 2.0 ** (np.arange(n_bins) / bins_per_octave)
    freq_labels = [f"{int(freqs[i])} Hz (mean)" for i in range(n_bins)] + \
                  [f"{int(freqs[i])} Hz (std)" for i in range(n_bins)]

    importances = rf.feature_importances_
    indices = np.argsort(importances)[-top_n:][::-1]
    top_features = [freq_labels[i] for i in indices]
    top_importances = importances[indices]

    sorted_idx = sorted(indices, key=lambda i: freqs[i % n_bins])
    sorted_features = [freq_labels[i] for i in sorted_idx]
    sorted_importances = [importances[i] for i in sorted_idx]

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.barh(sorted_features[::-1], sorted_importances[::-1], color='teal')
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} RF Features Sorted by Frequency")
    plt.tight_layout()
    plt.savefig("results/rf_features_sorted_by_frequency.png")
    plt.show()
    print("Saved: results/rf_features_sorted_by_frequency.png")

    plot_importance_binned_by_frequency(importances, freq_labels, bin_width=10,
        title="RF Importance Grouped by Frequency Bands (10Hz)",
        filename="results/rf_binned_by_frequency_band.png")

# Visualize permutation-based feature importance by frequency
def plot_permutation_importance_by_frequency(rf, X_test, y_test, top_n=40):
    print("Computing permutation importances...")
    result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    importances = result.importances_mean

    n_bins = X_test.shape[1] // 2
    bins_per_octave = 12
    fmin = 8.0
    freqs = fmin * 2.0 ** (np.arange(n_bins) / bins_per_octave)
    freq_labels = [f"{int(freqs[i])} Hz (mean)" for i in range(n_bins)] + \
                  [f"{int(freqs[i])} Hz (std)" for i in range(n_bins)]

    indices = np.argsort(importances)[-top_n:][::-1]
    top_features = [freq_labels[i] for i in indices]
    top_importances = importances[indices]

    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.barh(top_features[::-1], top_importances[::-1], color='slateblue')
    plt.xlabel("Permutation Importance")
    plt.title(f"Top {top_n} Permutation Importance Features by Frequency")
    plt.tight_layout()
    plt.savefig("results/permutation_importance_by_frequency.png")
    plt.show()
    print("Saved: results/permutation_importance_by_frequency.png")

    plot_importance_binned_by_frequency(top_importances, top_features, bin_width=10,
        title="Permutation Importance by Frequency Bands (10Hz)",
        filename="results/permutation_binned_by_frequency_band.png")

# Enhanced: Cumulative and Dynamic Binned Importance (≤ 100 Hz)
def plot_cumulative_and_dynamic_importance(rf_model, title_prefix="RF Feature Importance", save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    n_bins = rf_model.feature_importances_.shape[0] // 2
    bins_per_octave = 12
    fmin = 8.0
    freqs = fmin * 2.0 ** (np.arange(n_bins) / bins_per_octave)

    importances = rf_model.feature_importances_
    mean_importances = importances[:n_bins]
    std_importances = importances[n_bins:]
    total_importance = mean_importances + std_importances

    # Cumulative plot (≤ 100 Hz)
    mask = freqs <= 100
    sorted_idx = np.argsort(freqs[mask])
    sorted_freqs = freqs[mask][sorted_idx]
    sorted_importance = total_importance[mask][sorted_idx]
    cumulative = np.cumsum(sorted_importance)
    cumulative /= cumulative[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_freqs, cumulative, color='purple')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Cumulative Importance (0–1)")
    plt.title(f"{title_prefix}: Cumulative Importance")
    plt.grid(True)
    plt.xscale("log")
    plt.xlim(0, 100)
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cumulative_importance.png")
    plt.show()
    print(f"Saved: {save_dir}/cumulative_importance.png")

    # Dynamic Binning (~1/3 octave bands ≤ 100 Hz)
    max_freq = 100
    step_hz = 1 / 3
    num_bins = int(np.log2(max_freq / fmin) / step_hz) + 1
    dynamic_bins = [fmin * (2 ** (i * step_hz)) for i in range(num_bins)]

    bin_labels = []
    bin_values = []

    for i in range(len(dynamic_bins) - 1):
        low, high = dynamic_bins[i], dynamic_bins[i + 1]
        in_bin = (freqs >= low) & (freqs < high)
        if np.any(in_bin):
            avg_importance = np.mean(total_importance[in_bin])
            bin_values.append(avg_importance)
            bin_labels.append(f"{int(low)}–{int(high)} Hz")

    plt.figure(figsize=(12, 6))
    plt.bar(bin_labels, bin_values, color="darkgreen")
    plt.xlabel("Frequency Band (~1/3 Octave)")
    plt.ylabel("Avg Importance")
    plt.title(f"{title_prefix}: Dynamic Binned Importance (≤ 100 Hz)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/dynamic_binned_importance.png")
    plt.show()
    print(f"Saved: {save_dir}/dynamic_binned_importance.png")
