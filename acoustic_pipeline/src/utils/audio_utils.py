import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pathlib import Path
import warnings

from src.config import config


def bandpass_filter(signal, sr, lowcut=config.bandpass_lowcut,
                    highcut=config.bandpass_highcut, order=config.bandpass_order):
    try:
        nyq = 0.5 * sr
        low = lowcut / nyq
        high = highcut / nyq

        if not (0 < low < high < 1):
            raise ValueError(f"Invalid bandpass bounds: low={low:.4f}, high={high:.4f}, nyq={nyq}")

        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)

        # Compute power stats
        original_power = np.sum(signal ** 2)
        filtered_power = np.sum(filtered_signal ** 2)
        retained, removed = compute_power_stats(original_power, filtered_power)

        return filtered_signal, retained, removed

    except Exception as e:
        warnings.warn(f"Bandpass filter failed: {e}")
        return signal, 0.0, 100.0  # Return original with power removed



def extract_cqt_features_from_signal(signal, sr, 
                                     n_bins=config.n_bins,
                                     fmin=config.fmin,
                                     hop_length=512):
    try:
        cqt = librosa.cqt(signal, sr=sr, fmin=fmin, n_bins=n_bins, hop_length=hop_length)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        return np.mean(cqt_db, axis=1)
    except Exception as e:
        warnings.warn(f"CQT extraction failed: {e}")
        return np.zeros(n_bins)


def compute_power_stats(original_power, filtered_power):
    if original_power <= 0:
        return 0.0, 100.0
    retained = 100 * filtered_power / original_power
    removed = 100 - retained
    return retained, removed

def compare_average_spectrogram(original_signals, filtered_signals, sr, title="Average Spectrogram Comparison"):
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np

    if not original_signals or not filtered_signals:
        print("❌ No signals to compare.")
        return

    try:
        original_avg = np.mean([np.abs(librosa.stft(sig)) for sig in original_signals], axis=0)
        filtered_avg = np.mean([np.abs(librosa.stft(sig)) for sig in filtered_signals], axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        librosa.display.specshow(
            librosa.amplitude_to_db(original_avg, ref=np.max),
            sr=sr, y_axis='log', x_axis='time', ax=axes[0], cmap="magma"
        )
        axes[0].set_title("Original - Avg Spectrogram")

        librosa.display.specshow(
            librosa.amplitude_to_db(filtered_avg, ref=np.max),
            sr=sr, y_axis='log', x_axis='time', ax=axes[1], cmap="magma"
        )
        axes[1].set_title("Filtered - Avg Spectrogram")

        fig.suptitle(title)
        plt.tight_layout()

        from src.config import config
        plot_path = config.Paths.dnn_results / "plots" / "spectrogram_comparison.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        print(f"📊 Saved spectrogram comparison plot to {plot_path}")

    except Exception as e:
        print(f"❌ Error generating spectrograms: {e}")
