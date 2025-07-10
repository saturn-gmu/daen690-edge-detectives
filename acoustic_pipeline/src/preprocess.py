# scripts/audio_preprocessing_pipeline.py

import os
import re
import librosa
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed
from pathlib import Path
import matplotlib.pyplot as plt

# Fix for deprecated np.complex in librosa on newer numpy
np.complex = complex

from src.utils.audio_utils import extract_cqt_features_from_signal 
from src.utils.utils import compare_average_spectrogram
from src.config.config import sample_rate, bandpass_lowcut, bandpass_highcut, bandpass_order, folder_path, vesselnames, vessel
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, sr, lowcut, highcut, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
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

    y_filtered = bandpass_filter(y, sr, bandpass_lowcut, bandpass_highcut, bandpass_order)
    if y_filtered is None or len(y_filtered.shape) != 1:
        return None, None, f"Invalid filtered shape for {filename}"

    meta = {
        'filename': filename,
        'signal': y_filtered,
        'label': label,
        'vessel_labels': label_list,
        'target': target,
        'source_group': group
    }
    return meta, (y, y_filtered), None


def load_data_and_extract_features(folder_path, vesselnames, vessel,
                                   max_workers=4, n_jobs=4,
                                   default_n_bins=84, fmin=32.7, fmax=1000):
    data = []
    original_signals = []
    filtered_signals = []
    errors = []
    kept, skipped = 0, 0

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
                kept += 1
            elif error:
                errors.append(error)
                skipped += 1

    print(f"\n🔍 Filtering Report: Processed {len(wav_files)} files")
    print(f"✅ Kept: {kept} valid files")
    print(f"🚫 Skipped: {skipped} files")

    if original_signals and filtered_signals:
        compare_average_spectrogram(original_signals, filtered_signals, sample_rate)

    df = pd.DataFrame(data)
    if 'signal' not in df.columns or df.empty:
        print("❌ No valid signals processed.")
        return pd.DataFrame()

    print(f"⚙️ Extracting CQT features ({default_n_bins} bins, up to {fmax} Hz)...")

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
        print(f"⚠️ {len(errors)} files failed to load or process:")
        for msg in errors[:5]:
            print(" -", msg)
        if len(errors) > 5:
            print(" - ...")

    print(f"✅ Final dataset: {len(df)} valid samples with CQT features.")
    output_path = Path("artifacts/default/features_df.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_path)
    print(f"📦 Saved processed dataset to {output_path}")
    return df


if __name__ == "__main__":
    df = load_data_and_extract_features(folder_path, vesselnames, vessel)
