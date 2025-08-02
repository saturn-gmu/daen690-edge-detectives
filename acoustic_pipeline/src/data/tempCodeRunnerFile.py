
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import re

from src.data.pre_audio_filter import apply_bandpass_and_filter
from src.config import config
from src.config.config import Paths
from src.utils.data_utils import get_filtered_files, extract_features_parallel

def assign_labels_from_path(file_path):
    match = re.search(r"(SpeedBoat|QianDao|KaiYuan|No7|UUV|GreenCity|TheEarl|Cargo|FishBoat|Unknown|TheKnight|WorkShip|ArtificialSignals|BigPassengerShip|PoliceBoat|MotorBoat|Car|CivilianBoats|No5|Helicopter)", str(file_path))
    if match:
        return match.group(1)
    return "Unknown"

def get_or_generate_features(features_path=None, ranked_path=None, save_path=None):
    features_path = Path(features_path or Paths.features_parquet)
    ranked_path = Path(ranked_path or Paths.ranked_features_csv)

    if features_path.exists():
        print(f"📥 Loading features from: {features_path}")
        df = pd.read_parquet(features_path)
    else:
        source_dir = Path(config.folder_path)
        print(f"⚙️ No features found — generating from: {source_dir}")
        audio_files = get_filtered_files(source_dir)

        features = extract_features_parallel(
            file_paths=audio_files,
            sr=config.sample_rate,
            n_bins=config.n_bins,
            fmin=config.fmin,
            hop_length=config.cqt_hop_length,
        )

        paths, vessels, targets = [], [], []
        for f in audio_files:
            vessel = assign_labels_from_path(f)
            paths.append(str(f))
            vessels.append(vessel)
            targets.append(1 if config.vessel in vessel else 0)

        df = pd.DataFrame(features)
        df["path"] = paths
        df["vessel"] = vessels
        df["target"] = targets

        features_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(features_path)
        print(f"💾 Saved features to: {features_path}")

        if ranked_path:
            ranked_path.parent.mkdir(parents=True, exist_ok=True)
            ranked = [str(c) for c in df.columns if isinstance(c, (str, int)) and str(c).isdigit()]
            with open(ranked_path, "w") as f:
                f.write("\\n".join(ranked))
            print(f"🧠 Saved ranked features to: {ranked_path}")

    feature_cols = [col for col in df.columns if str(col).isdigit()]
    print(f"✅ Final dataset: {len(df)} samples | {len(feature_cols)} features")
    return df, feature_cols

