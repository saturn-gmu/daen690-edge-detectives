# filter_short_audio.py

import os
import librosa
import shutil

MIN_LENGTH = 256  # minimum samples required (match your n_fft)
SOURCE_DIR = "/Users/christopherhurtig/Desktop/QiandaoEar22"
OUTPUT_DIR = "/Users/christopherhurtig/Desktop/QiandaoEar22_filtered"

os.makedirs(OUTPUT_DIR, exist_ok=True)

kept, skipped = 0, 0

for filename in os.listdir(SOURCE_DIR):
    if not filename.lower().endswith(".wav"):
        continue

    path = os.path.join(SOURCE_DIR, filename)
    try:
        y, sr = librosa.load(path, sr=22050)
        if len(y) >= MIN_LENGTH:
            shutil.copy2(path, os.path.join(OUTPUT_DIR, filename))
            kept += 1
        else:
            skipped += 1
    except Exception as e:
        print(f"❌ Failed to load {filename}: {e}")
        skipped += 1

print(f"✅ Completed filtering. Kept {kept} files. Skipped {skipped}.")
