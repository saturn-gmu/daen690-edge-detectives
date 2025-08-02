#preprocess.py

# === Imports ===
import os, re, gc
import librosa                  
import numpy as np               
import pandas as pd              
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path         
from concurrent.futures import ThreadPoolExecutor, as_completed  
from joblib import Parallel, delayed                              
from scipy.signal import butter, filtfilt                        
from src.utils.audio_utils import bandpass_filter


# Patch to support older librosa using np.complex
np.complex = complex

# === Local module imports ===
from src.utils.audio_utils import extract_cqt_features_from_signal, compare_average_spectrogram
from src.config.config import (
    sample_rate, bandpass_lowcut, bandpass_highcut, bandpass_order,
    folder_path, vesselnames, vessel
)

