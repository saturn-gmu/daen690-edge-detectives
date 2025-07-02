# This utility module provides simple timing functions to measure execution time and signal diagnostics.

import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

# Start a timer and return the current time
def start_timer():
    return time.time()

# Calculate and print the elapsed time since `start_time`
def end_timer(start_time):
    end = time.time()
    print(f"\n⏱️ Total Execution Time: {end - start_time:.2f} seconds")

# Compare average spectrograms of original vs. filtered signals
def compare_average_spectrogram(originals, filtereds, sr, title="Average Spectrogram Comparison"):
    # Normalize and truncate signals
    min_len = min(min(len(x) for x in originals), min(len(x) for x in filtereds))
    originals = [x[:min_len] for x in originals]
    filtereds = [x[:min_len] for x in filtereds]

    # Compute spectrograms and accumulate
    def avg_spectrogram(signals):
        specs = []
        for x in signals:
            f, t, Sxx = spectrogram(x, fs=sr, nperseg=512, noverlap=256)
            specs.append(10 * np.log10(Sxx + 1e-10))
        return f, t, np.mean(specs, axis=0)

    f, t, avg_orig = avg_spectrogram(originals)
    _, _, avg_filt = avg_spectrogram(filtereds)

    # Plot side-by-side spectrograms
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    im1 = axs[0].pcolormesh(t, f, avg_orig, shading='gouraud', cmap='magma')
    axs[0].set_title("Original - Avg Spectrogram")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im1, ax=axs[0], label="dB")

    im2 = axs[1].pcolormesh(t, f, avg_filt, shading='gouraud', cmap='magma')
    axs[1].set_title("Filtered - Avg Spectrogram")
    axs[1].set_xlabel("Time (s)")
    fig.colorbar(im2, ax=axs[1], label="dB")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
