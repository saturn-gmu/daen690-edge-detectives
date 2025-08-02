#data_utils.py


from pathlib import Path
import os
import numpy as np
import librosa
from joblib import Parallel, delayed
from src.utils.audio_utils import extract_cqt_features_from_signal

def get_filtered_files(source_dir):
    source_dir = Path(source_dir)
    audio_files = list(source_dir.rglob("*.wav"))
    print(f"✅ Found {len(audio_files)} .wav files in {source_dir}")
    return audio_files

def extract_features_parallel(file_paths, sr, n_bins, fmin, hop_length):
    def process_file(file_path):
        try:
            signal, _ = librosa.load(file_path, sr=sr)
            features = extract_cqt_features_from_signal(signal, sr, n_bins, fmin, hop_length)
            return features
        except Exception as e:
            print(f"❌ Failed to extract features from {file_path}: {e}")
            return None

    results = Parallel(n_jobs=-1)(delayed(process_file)(f) for f in file_paths)
    return [r for r in results if r is not None]
