
# This module loads .wav audio files from a specified directory,
# applies preprocessing including bandpass filtering and feature extraction,
# and builds a labeled dataset.

import os
import re
import librosa
import numpy as np
import pandas as pd
from src.preprocess import extract_cqt_features_from_signal, bandpass_filter
from src.utils import compare_average_spectrogram

# Load .wav files, apply bandpass filter, extract vessel labels, and generate features
def load_data_and_extract_features(folder_path, vesselnames, vessel):
    from src.config import sample_rate, bandpass_lowcut, bandpass_highcut, bandpass_order

    data = []
    original_signals = []
    filtered_signals = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".wav"):
            path = os.path.join(folder_path, filename)
            try:
                y, sr = librosa.load(path, sr=sample_rate)
            except:
                continue

            matches = re.findall(vesselnames, filename)
            label_list = sorted(set(matches))
            label = " ".join(label_list)
            target = 1 if vessel in label_list else 0

            # Determine grouping
            if vessel in label_list and len(label_list) == 1:
                group = "only"
            elif vessel in label_list:
                group = "contains"
            else:
                group = "not"

            # Apply bandpass filter
            y_filtered = bandpass_filter(y, sr, bandpass_lowcut, bandpass_highcut, bandpass_order)

            if y_filtered is None or len(y_filtered.shape) != 1:
                continue

            original_signals.append(y)
            filtered_signals.append(y_filtered)

            data.append({
                'filename': filename,
                'signal': y_filtered,
                'label': label,
                'vessel_labels': label_list,
                'target': target,
                'source_group': group
            })

    # Compare average spectrogram of filtered vs original signals
    if original_signals and filtered_signals:
        compare_average_spectrogram(original_signals, filtered_signals, sample_rate)

    df = pd.DataFrame(data)
    if 'signal' not in df.columns:
        return pd.DataFrame()

    df['features'] = df['signal'].apply(extract_cqt_features_from_signal)
    return df[df['features'].notnull()].copy()
