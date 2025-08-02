
# 🔊 DNN-Based Underwater Vessel Classification Pipeline

This repository provides a full audio classification pipeline for detecting and classifying underwater vessels using deep learning. The pipeline supports preprocessing, feature extraction (CQT), model training (DNN), pruning, quantization (INT8), evaluation, and visualization.

---

## 📁 Directory Structure

```
DNN_Model/
├── README.md
├── requirements.txt
├── scripts/                         # All executable pipelines and CLI scripts
│   ├── dnn_initial.py              # Float32 DNN training
│   ├── dnn_summary.py              # Evaluation and plotting of DNN
│   ├── optimized_ptq.py            # Pruned + quantized DNN evaluation
│   ├── ptq_pipeline.py             # PTQ + INT8 wrapper
│   ├── plot_ptq_summary.py         # Generate summary plots
│   ├── run_ablation.py             # Feature ablation (Top-K sweep)
│   └── quantization.py             # Legacy converter (optional)
│
├── artifacts/                      # Features, model weights, inference artifacts
│   ├── features_df.pkl
│   └── ablation/
│       ├── ranked_features.csv
│       └── masks/ (Top-K feature masks)
│
├── results/                        # Output CSVs, plots, and logs
│   ├── dnn/
│   ├── ptq/
│   └── plots/
│
├── src/                            # Core logic modules
│   ├── config/                     # Global config.py
│   ├── data/                       # Filtering, CQT extraction, preprocessing
│   ├── models/                     # DNN, RF model definitions
│   ├── evaluation/                 # Metric, ROC, PR evaluation
│   ├── quantization/              # INT8 + pruning support
│   ├── ablation/                  # Feature masking + importance
│   └── utils/                     # Helpers: audio, I/O, timing, visualization
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourname/DNN_Model.git
cd DNN_Model
```

2. **Install required dependencies**
```bash
pip install -r requirements.txt
```

3. **Edit Configuration**
Open `src/config/config.py` to set:
- `folder_path`: Path to your filtered .wav dataset
- `vessel`: Vessel label to target (e.g., `"No7"`, `"SpeedBoat"`)
- Other values: sample rate, CQT parameters, and centralized `Paths`

---

## 🔍 Pipeline Overview

### 🛠️ 1. Preprocessing & Feature Extraction

```bash
python scripts/dnn_initial.py
```
- Applies bandpass filter to raw audio
- Extracts CQT features using `librosa`
- Saves `.pkl` feature file and `.csv` of ranked features

### 🧠 2. DNN Training

Trains the base DNN on Top-K features (5 to 65):

```bash
python scripts/dnn_initial.py
```
- Trains `.h5` models
- Logs metrics to `results/dnn/pipeline_metrics.csv`
- Exports `.npz` inference files

### ✂️ 3. Pruning + Quantization

To apply pruning and INT8 quantization:

```bash
python scripts/optimized_ptq.py
```
- Uses TensorFlow Model Optimization Toolkit
- Benchmarks pruned float32 and INT8 `.tflite` models

To perform standalone PTQ evaluation:

```bash
python scripts/ptq_pipeline.py
```

---

### 📈 4. Visualization

Generate plots summarizing accuracy, inference time, model size:

```bash
python scripts/plot_ptq_summary.py
```

To view DNN-specific metrics, ROC, and PR curves:

```bash
python scripts/dnn_summary.py
```

---

## 🧪 Feature Ablation (Top-K Sweep)

```bash
python scripts/run_ablation.py
```

- Applies Top-K masks from `artifacts/ablation/masks/`
- Evaluates impact on RF and DNN performance
- Plots saved to `results/plots/ablation/`

---

## 🛠 Configuration Details

`src/config/config.py` contains:
- Audio preprocessing: `sample_rate`, `bandpass_lowcut`, `bandpass_highcut`
- Feature extraction: `n_bins`, `cqt_hop_length`, `fmin`
- Dataset filtering: `min_sample_length`, `vessel`, `vesselnames`
- Path management: `Paths` class defines all save/load locations centrally

Use `Paths.make_dirs()` to auto-create directories.

---

## 🧼 Troubleshooting

- ❗ `KeyError: 'target'`: Check your `vessel` string in `config.py`
- ❗ `No module named 'src'`: Run from project root or set `PYTHONPATH=.`
- ❗ No `.wav` found: Verify your `folder_path` exists and is non-empty

---

## 📌 Acknowledgments

This project uses:
- TensorFlow (pruning + quantization)
- Librosa (CQT feature extraction)
- Scikit-learn (Random Forest, metrics)
- Seaborn + matplotlib (plotting)

Supports low-power inference through `.tflite` export.
