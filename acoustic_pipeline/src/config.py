#config.py
# This configuration file contains parameters for the DNN model, data loading, and feature extraction.  

# Dataset location and vessel filtering
folder_path = "/Users/christopherhurtig/Desktop/QiandaoEar22_filtered"
vesselnames = "SpeedBoat|QianDao|KaiYuan|No7|UUV|GreenCity|TheEarl|Cargo|FishBoat|Unknown|TheKnight|WorkShip|ArtificialSignals|BigPassengerShip|PoliceBoat|MotorBoat|Car|CivilianBoats|No5|Helicopter"
vessel = "No7"  # Default vessel for training, can be overridden

# Bandpass filter configuration
bandpass_low = 20
bandpass_high = 1000
bandpass_lowcut = bandpass_low
bandpass_highcut = bandpass_high
bandpass_order = 5

# CQT feature configuration
sample_rate = 22050
bins_per_octave = 36     # More resolution per octave
n_bins = 108             # 3 full octaves at 36 bins/octave = covers up to ~1024 Hz
fmin = 8.0               # Keep minimum frequency the same

# DNN training configuration
epochs = 15
batch_size = 32





