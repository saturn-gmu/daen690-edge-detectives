from pathlib import Path

# === Basic Audio Dataset Settings ===
sample_rate = 22050
bandpass_lowcut = 20
bandpass_highcut = 1000
bandpass_order = 5

# CQT settings
bins_per_octave = 36
n_bins = 108
fmin = 8.0
cqt_bins = 84           # total number of frequency bins
cqt_fmin = 32.7         # minimum frequency (Hz), typically C1
cqt_hop_length = 512    # hop length in samples

# DNN training settings
epochs = 15
batch_size = 32

# Dataset filtering
min_sample_length = 256

# Vessel selection
folder_path = "/Users/christopherhurtig/Desktop/QiandaoEar22_filtered"
raw_audio_dir = "/Users/christopherhurtig/Desktop/QiandaoEar22"
filtered_audio_dir = folder_path
vesselnames = "SpeedBoat|QianDao|KaiYuan|No7|UUV|GreenCity|TheEarl|Cargo|FishBoat|Unknown|TheKnight|WorkShip|ArtificialSignals|BigPassengerShip|PoliceBoat|MotorBoat|Car|CivilianBoats|No5|Helicopter"
vessel = "No7"

# === Centralized Path Management ===
class Paths:
    BASE = Path("/Users/christopherhurtig/Desktop/DNN_Model")

    # Artifacts and data
    features_parquet = BASE / "artifacts/features_df.parquet"
    ranked_features_csv = BASE / "artifacts/ablation/ranked_features.csv"
    feature_masks = BASE / "artifacts/ablation/masks"
    ablation_threshold_masks = BASE / "artifacts/ablation/masks"
    data_filtered = Path("/Users/christopherhurtig/Desktop/QiandaoEar22_filtered")

    # Model outputs
    h5_models_dir = BASE / "results/dnn"
    ptq_models_dir = BASE / "artifacts/ptq_models"
    tflite_models_dir = BASE / "artifacts/tflite"

    # Results CSVs
    dnn_results = BASE / "results/dnn"
    ptq_results = BASE / "results/ptq"
    ablation_results = BASE / "results/ablation"
    pipeline_results = BASE / "results/pipeline"

    # Plot folders
    plot_base = BASE / "results/plots"
    filter_plot_dir = plot_base / "filter_analysis"
    dnn_plot_dir = plot_base / "dnn_training"
    ptq_plot_dir = plot_base / "ptq_comparison"
    ablation_plot_dir = plot_base / "ablation"
    pipeline_plots = plot_base / "pipeline"
    dnn_training_plots = plot_base / "dnn_training"
    roc_curves = plot_base / "roc"
    pr_curves = plot_base / "pr"
    feature_importance = plot_base / "feature_importance"
    val_vs_test = plot_base / "val_vs_test"
    summary_plots = plot_base / "summary"

    @classmethod
    def make_dirs(cls):
        for attr in dir(cls):
            if not attr.startswith("_"):
                val = getattr(cls, attr)
                if isinstance(val, Path):
                    val.parent.mkdir(parents=True, exist_ok=True)

# Top-K sweep settings
class TopK:
    start = 5
    end = 65
    step = 5

# Expose main features path as string
FEATURES_PKL_PATH = str(Paths.features_parquet)
