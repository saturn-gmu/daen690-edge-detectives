# === src/data/generate_features.py ===

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import librosa
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier

from src.config.config import (
    sample_rate, n_bins, fmin, cqt_hop_length,
    folder_path, vessel,
)
from src.config.config import Paths
from src.utils.audio_utils import extract_cqt_features_from_signal
from src.utils.data_utils import get_filtered_files
from src.data.pre_audio_filter import bandpass_filter, compute_power_stats


def extract_features_parallel(file_paths):
    def process_file(file_path):
        try:
            signal, _ = librosa.load(file_path, sr=sample_rate)
            filtered_signal = bandpass_filter(signal, sample_rate)

            original_power = np.sum(signal ** 2)
            filtered_power = np.sum(filtered_signal ** 2)
            retained, removed = compute_power_stats(original_power, filtered_power)

            features = extract_cqt_features_from_signal(
                filtered_signal, sample_rate, n_bins, fmin, cqt_hop_length
            )

            return features, file_path, retained, removed

        except Exception as e:
            print(f"❌ Failed: {file_path} — {e}")
            return None, None, None, None

    results = Parallel(n_jobs=-1)(delayed(process_file)(f) for f in tqdm(file_paths))
    return [(f, p, r, m) for f, p, r, m in results if f is not None]


def get_or_generate_features(features_path=None, ranked_path=None):
    features_path = Path(features_path or Paths.features_parquet)
    ranked_path = Path(ranked_path or Paths.ranked_features_csv)

    if features_path.exists():
        print(f"\n📅 Loading features from: {features_path}")
        df = pd.read_parquet(features_path)
    else:
        print(f"\n⚙️ No features found — generating from: {folder_path}")
        files = get_filtered_files(folder_path)
        print(f"🔍 Filtering {len(files)} files from {folder_path}")
        results = extract_features_parallel(files)

        data = []
        for features, path, retained, removed in results:
            row = {f"f_{i}": v for i, v in enumerate(features)}
            row["path"] = str(path)
            row["power_retained_percent"] = retained
            row["power_removed_percent"] = removed
            row["target"] = 1 if vessel in str(path) else 0
            data.append(row)

        df = pd.DataFrame(data)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(features_path)
        print(f"💾 Saved features to: {features_path}")

    feature_cols = [col for col in df.columns if col.startswith("f_")]
    print(f"\n✅ Final dataset: {len(df)} samples | {len(feature_cols)} features")

    # Early exit if ranked features already exist
    if ranked_path.exists():
        print(f"📄 Using existing ranked features from: {ranked_path}")
        ranked_df = pd.read_csv(ranked_path)
        if "feature" in ranked_df.columns:
            ranked_features = ranked_df["feature"].tolist()
        else:
            ranked_features = [f"f_{int(i)}" for i in ranked_df.iloc[:, 0]]
        return df, ranked_features

    # Train RF to rank features
    ranked_features = []
    if "target" in df.columns:
        try:
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            rf.fit(df[feature_cols], df["target"])
            importances = rf.feature_importances_
            ranked_features = [
                f for _, f in sorted(zip(importances, feature_cols), reverse=True)
            ]
            ranked_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"feature": ranked_features}).to_csv(ranked_path, index=False)
            print(f"🏷️  Saved ranked features to: {ranked_path}")
        except Exception as e:
            print(f"⚠️ Failed to compute ranked features: {e}")
    else:
        print("⚠️ No 'target' column found. Ranked features not computed.")

    return df, ranked_features
