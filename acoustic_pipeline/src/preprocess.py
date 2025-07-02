# preprocess.py
# This module handles preprocessing of audio signals, including normalization,
# silence trimming, bandpass filtering, and CQT feature extraction.

import numpy as np
import librosa
from scipy.signal import butter, lfilter
from src.config import bandpass_lowcut, bandpass_highcut, bandpass_order

# Applies a Butterworth bandpass filter to the input signal
def bandpass_filter(data, sr, lowcut=bandpass_lowcut, highcut=bandpass_highcut, order=bandpass_order):
    nyq = 0.5 * sr  # Nyquist frequency
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return lfilter(b, a, data)

# Extracts Constant-Q Transform (CQT) features from a mono audio signal
def extract_cqt_features_from_signal(y, sr=22050, n_bins=84):
    try:
        # Convert to mono if stereo
        y = librosa.to_mono(y)

        # Normalize the signal to [-1, 1] range
        y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y

        # Trim leading/trailing silence
        y, _ = librosa.effects.trim(y, top_db=30)

        # Apply bandpass filter to isolate relevant frequency range
        y = bandpass_filter(y, sr)

        # Compute the magnitude of the Constant-Q Transform
        cqt = np.abs(librosa.cqt(y, sr=sr, fmin=8.0, n_bins=n_bins, bins_per_octave=12))

        # Return mean and standard deviation across time for each frequency bin
        return np.concatenate([np.mean(cqt, axis=1), np.std(cqt, axis=1)])
    except:
        # Return None for signals that cannot be processed
        return None