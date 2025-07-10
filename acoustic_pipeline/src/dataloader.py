# Import required libraries for audio processing, concurrency, and data manipulation
import os
import re
import librosa
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed  # for multithreaded processing
from joblib import Parallel, delayed  # for parallel feature extraction

# Import project-specific utilities
from src.utils.audio_utils import extract_cqt_features_from_signal
from src.utils.utils import compare_average_spectrogram
from src.config.config import sample_rate, bandpass_lowcut, bandpass_highcut, bandpass_order
from scipy.signal import butter, filtfilt  # for bandpass filter

# === Bandpass filter utility ===
def bandpass_filter(signal, sr, lowcut, highcut, order=5):
    nyq = 0.5 * sr  # Nyquist frequency (half the sample rate)
    low = lowcut / nyq  # Normalize lowcut
    high = highcut / nyq  # Normalize highcut
    b, a = butter(order, [low, high], btype='band')  # Create filter coefficients
    return filtfilt(b, a, signal)  # Apply the filter (zero-phase distortion)

# === Process a single WAV file ===
def process_file(filename, folder_path, vesselnames, vessel):
    path = os.path.join(folder_path, filename)  # Build full file path

    try:
        y, sr = librosa.load(path, sr=sample_rate)  # Load audio signal at target sample rate
    except Exception as e:
        return None, None, f"Failed to load {filename}: {e}"  # Skip if loading fails

    matches = re.findall(vesselnames, filename)  # Extract vessel labels from filename
    label_list = sorted(set(matches))  # Remove duplicates and sort
    label = " ".join(label_list)  # Create a label string
    target = 1 if vessel in label_list else 0  # Binary target label (1 if desired vessel present)

    # Group categorization: "only", "contains", or "not"
    group = "only" if vessel in label_list and len(label_list) == 1 else (
        "contains" if vessel in label_list else "not"
    )

    # Apply bandpass filter
    y_filtered = bandpass_filter(y, sr, bandpass_lowcut, bandpass_highcut, bandpass_order)
    if y_filtered is None or len(y_filtered.shape) != 1:
        return None, None, f"Invalid filtered shape for {filename}"

    # Metadata dictionary for one file
    meta = {
        'filename': filename,
        'signal': y_filtered,
        'label': label,
        'vessel_labels': label_list,
        'target': target,
        'source_group': group
    }
    return meta, (y, y_filtered), None  # Also return original and filtered signals for diagnostics

# === Main data loading + feature extraction ===
def load_data_and_extract_features(folder_path, vesselnames, vessel,
                                   max_workers=4, n_jobs=4,
                                   default_n_bins=84, fmin=32.7, fmax=1000):
    data = []  # Stores metadata dictionaries
    original_signals = []  # Original unfiltered audio (for spectrogram comparison)
    filtered_signals = []  # Filtered audio signals
    errors = []  # Errors encountered while loading

    # Get all .wav files in the folder
    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]

    # Multithreaded preprocessing of audio files
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file, f, folder_path, vesselnames, vessel): f
            for f in wav_files
        }

        # Collect results as they complete
        for future in as_completed(futures):
            result, signals, error = future.result()
            if result:
                data.append(result)
                if signals:
                    original_signals.append(signals[0])  # original
                    filtered_signals.append(signals[1])  # filtered
            elif error:
                errors.append(error)

    # Optional visualization of average spectrogram before and after filtering
    if original_signals and filtered_signals:
        compare_average_spectrogram(original_signals, filtered_signals, sample_rate)

    # Convert metadata into a DataFrame
    df = pd.DataFrame(data)
    if 'signal' not in df.columns or df.empty:
        print("❌ No valid signals processed.")
        return pd.DataFrame()

    print(f"⚙️ Extracting CQT features (max {fmax} Hz, {default_n_bins} bins) using {n_jobs} workers...")

    # Wrap CQT extraction in a try-except to prevent crashes
    def safe_cqt(sig):
        try:
            return extract_cqt_features_from_signal(sig, sr=sample_rate,
                                                    n_bins=default_n_bins,
                                                    fmin=fmin, fmax=fmax)
        except Exception:
            return None

    # Parallel CQT extraction for each signal
    df['features'] = Parallel(n_jobs=n_jobs)(
        delayed(safe_cqt)(sig) for sig in df['signal']
    )
    df = df[df['features'].notnull()].copy()  # Drop failed extractions

    # Report errors (limit to 5 shown)
    if errors:
        print(f"⚠️ {len(errors)} files failed to load or process.")
        for msg in errors[:5]:
            print(" -", msg)
        if len(errors) > 5:
            print(" - ...")

    print(f"✅ Loaded {len(df)} valid samples with CQT features.")
    return df  # Return final DataFrame with features and metadata

