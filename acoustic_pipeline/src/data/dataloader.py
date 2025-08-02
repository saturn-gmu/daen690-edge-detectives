# src/data/dataloader.py
# === Full Preprocessing & Feature Extraction Script with Histograms and Debug Guards ===

import os
import re
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt

from src.utils.audio_utils import extract_cqt_features_from_signal, compare_average_spectrogram
from src.config.config import sample_rate, bandpass_lowcut, bandpass_highcut, bandpass_order

def bandpass_filter(signal, sr, lowcut, highcut, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    if not (0 < low < 1) or not (0 < high < 1):
        raise ValueError(f"Critical frequencies out of bounds: low={low}, high={high}, sr={sr}")
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def process_file(filename, folder_path, vesselnames, vessel):
    path = os.path.join(folder_path, filename)
    try:
        y, sr = librosa.load(path, sr=sample_rate)
    except Exception as e:
        return None, None, f"Failed to load {filename}: {e}"

    matches = re.findall(vesselnames, filename)
    label_list = sorted(set(matches))
    label = " ".join(label_list)
    target = 1 if vessel in label_list else 0
    group = "only" if vessel in label_list and len(label_list) == 1 else (
        "contains" if vessel in label_list else "not")

    try:
        y_filtered = bandpass_filter(y, sr, bandpass_lowcut, bandpass_highcut, bandpass_order)
    except Exception as e:
        return None, None, f"Filtering error in {filename}: {e}"

    if y_filtered is None or len(y_filtered.shape) != 1:
        return None, None, f"Invalid filtered shape for {filename}"

    power_pre = np.sum(y ** 2)
    power_post = np.sum(y_filtered ** 2)

    retained = 100.0 * power_post / power_pre if power_pre > 0 else 0.0
    removed = 100.0 - retained

    meta = {
        'filename': filename,
        'signal': y_filtered,
        'label': label,
        'vessel_labels': label_list,
        'target': target,
        'source_group': group,
        'pre_filter_power': power_pre,
        'post_filter_power': power_post,
        'power_retained_percent': retained,
        'power_removed_percent': removed
    }

    return meta, (y, y_filtered), None

def plot_combined_power_histogram(df, column, title, filename):
    if column not in df.columns or df[column].isnull().all():
        print(f"❌ Skipping histogram — column '{column}' missing or empty.")
        return
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, hue="source_group", bins=30,
                 kde=True, element="step", stat="density", common_norm=False)
    plt.title(title)
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("DNN_Model/results/plots", exist_ok=True)
    plt.savefig(f"DNN_Model/results/plots/{filename}")
    plt.close()
    print(f"📊 Saved plot to DNN_Model/results/plots/{filename}")

def load_data_and_extract_features(folder_path, vesselnames, vessel,
                                   max_workers=4, n_jobs=4,
                                   default_n_bins=84, fmin=32.7, fmax=1000):
    data = []
    original_signals = []
    filtered_signals = []
    errors = []

    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file, f, folder_path, vesselnames, vessel): f
            for f in wav_files
        }
        for future in as_completed(futures):
            result, signals, error = future.result()
            if result:
                data.append(result)
                if signals:
                    original_signals.append(signals[0])
                    filtered_signals.append(signals[1])
            elif error:
                errors.append(error)

    if original_signals and filtered_signals:
        compare_average_spectrogram(original_signals, filtered_signals, sample_rate)

    df = pd.DataFrame(data)

    if 'signal' not in df.columns or df.empty:
        print("❌ No valid signals processed.")
        return pd.DataFrame()

    print(f"⚙️ Extracting CQT features (max {fmax} Hz, {default_n_bins} bins)...")

    def safe_cqt(sig):
        try:
            return extract_cqt_features_from_signal(sig, sr=sample_rate,
                                                    n_bins=default_n_bins,
                                                    fmin=fmin, fmax=fmax)
        except Exception:
            return None

    df['features'] = Parallel(n_jobs=n_jobs)(
        delayed(safe_cqt)(sig) for sig in df['signal']
    )

    df = df[df['features'].notnull()].copy()

    if errors:
        print(f"⚠️ {len(errors)} files failed.")
        for msg in errors[:5]:
            print(" -", msg)
        if len(errors) > 5:
            print(" - ...")

    print(f"✅ Final dataset: {len(df)} valid samples with features.")

    total_pre_energy = df["pre_filter_power"].sum()
    total_post_energy = df["post_filter_power"].sum()
    total_removed_fraction = (100.0 * (total_pre_energy - total_post_energy) / total_pre_energy) if total_pre_energy else 0
    mean_retained = df["power_retained_percent"].mean()
    mean_removed = df["power_removed_percent"].mean()

    print("\n📊 Bandpass Filter Summary:")
    print(f"   • Average power retained: {mean_retained:.2f}%")
    print(f"   • Average power removed: {mean_removed:.2f}%")
    print(f"   • Total energy removed (across dataset): {total_removed_fraction:.2f}%")

    group_summary = df.groupby("source_group")[["power_retained_percent", "power_removed_percent"]].mean()
    print("\n📈 Filter impact by source group:")
    print(group_summary.round(2).to_string())

    plot_combined_power_histogram(df, "power_retained_percent", "Power Retained After Bandpass Filter", "power_retained_histogram.png")
    plot_combined_power_histogram(df, "power_removed_percent", "Power Removed by Bandpass Filter", "power_removed_histogram.png")

    return df