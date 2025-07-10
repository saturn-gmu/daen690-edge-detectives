# src/utils/audio_utils.py
import numpy as np
import librosa


def extract_cqt_features_from_signal(signal, sr, n_bins=84, fmin=32.7, fmax=1000):
    """
    Extracts averaged CQT (Constant-Q Transform) features from an audio signal.

    Parameters:
        signal (np.ndarray): The raw audio signal.
        sr (int): Sample rate of the signal.
        n_bins (int): Number of frequency bins.
        fmin (float): Minimum frequency.
        fmax (float): Maximum frequency.

    Returns:
        np.ndarray: Mean CQT magnitude values across time.
    """
    cqt = librosa.cqt(signal, sr=sr, fmin=fmin, n_bins=n_bins)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    return np.mean(cqt_db, axis=1)
