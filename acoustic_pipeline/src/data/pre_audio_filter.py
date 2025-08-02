# === src/data/pre_audio_filter.py ===

import numpy as np
from scipy.signal import butter, filtfilt
import warnings
from src.config import config

def bandpass_filter(signal, sr, 
                    lowcut=config.bandpass_lowcut, 
                    highcut=config.bandpass_highcut, 
                    order=config.bandpass_order):
    try:
        nyq = 0.5 * sr
        low = lowcut / nyq
        high = highcut / nyq

        if not (0 < low < high < 1):
            raise ValueError(f"Invalid bandpass bounds: low={low:.4f}, high={high:.4f}, nyq={nyq}")

        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    except Exception as e:
        warnings.warn(f"Bandpass filter failed: {e}")
        return signal  # Return original if filtering fails


def compute_power_stats(original_power, filtered_power):
    if original_power <= 0:
        return 0.0, 100.0
    retained = 100 * filtered_power / original_power
    removed = 100 - retained
    return retained, removed
