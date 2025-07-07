import os
import re
import numpy as np
import matplotlib.pyplot as plt
import librosa

from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    matthews_corrcoef
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

#defining constants from config

SAMPLE_RATE = 22050
BANDPASS_LOWCUT = 50
BANDPASS_HIGHCUT = 1000
BANDPASS_ORDER = 5

# functions for filtering and feature extraction

def bandpass_filter(signal, sr, lowcut, highcut, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def extract_features(signal, sr=SAMPLE_RATE):
    try:
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None

#loading datat & preprocessing 

def load_data(folder_path, vessel_keyword):
    X, y, filenames = [], [], []
    for fname in os.listdir(folder_path):
        if fname.endswith(".wav"):
            label = 1 if vessel_keyword in fname else 0
            try:
                filepath = os.path.join(folder_path, fname)
                y_raw, sr = librosa.load(filepath, sr=SAMPLE_RATE)
                y_filtered = bandpass_filter(y_raw, sr, BANDPASS_LOWCUT, BANDPASS_HIGHCUT, BANDPASS_ORDER)
                feat = extract_features(y_filtered, sr)
                if feat is not None:
                    X.append(feat)
                    y.append(label)
                    filenames.append(fname)
            except Exception as e:
                print(f"Failed to process {fname}: {e}")
    return np.array(X), np.array(y), filenames

#preprocessing, training & evaluation

if __name__ == "__main__":
    data_folder = r"C:\Users\lanlo\Desktop\DAEN690\sprint 3\QiandaoEar22\QiandaoEar22"
    keyword = "No7"  # Change this to your vessel of interest

    print(" Loading data...")
    X, y, filenames = load_data(data_folder, keyword)

    print(" Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(" Training Random Forest...")
    random_forest_model = RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_leaf=5,class_weight='balanced', random_state=42)
    random_forest_model.fit(X_train_scaled, y_train)

    print(" Evaluation:")
    y_pred = random_forest_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    
ConfusionMatrixDisplay.from_estimator(random_forest_model, X_test_scaled, y_test)
plt.title("Test Set Confusion Matrix")
plt.show()

#visualizing rf:
    
feature_names = X.columns if hasattr(X, "columns") else [f"Feature {i}" for i in range(X.shape[1])]
importances = random_forest_model.feature_importances_
indices = np.argsort(importances)[::-1]  # descending order

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

#checking for overfitting 

print("Train accuracy:", random_forest_model.score(X_train_scaled, y_train))

print("Test accuracy:", random_forest_model.score(X_test_scaled, y_test))


#cross-validation for overfitting 
random_forest_model = RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_leaf=5, class_weight='balanced',random_state=42)

# StratifiedKFold preserves class distribution in splits
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Run cross-validation with accuracy scoring
scores = cross_val_score(random_forest_model, X, y, cv=cv, scoring='accuracy')

print("Cross-validation accuracy scores:", scores)
print("Mean accuracy:", np.mean(scores))
print("Standard deviation:", np.std(scores))


#cross validtaion with multiple metrics 

# Create a pipeline that scales and then classifies
pipeline = make_pipeline(StandardScaler(), random_forest_model)

scoring = ['accuracy', 'precision', 'recall', 'f1']

cv_results = cross_validate(pipeline, X, y, cv=5, scoring=scoring)

for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"{metric.capitalize()} scores: {scores}")
    print(f"Mean {metric}: {np.mean(scores):.4f}")
    print(f"Std {metric}: {np.std(scores):.4f}")
    print()

#roc curve 

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
plt.figure(figsize=(8,6))

for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    pipeline.fit(X[train_idx], y[train_idx])
    y_proba = pipeline.predict_proba(X[test_idx])[:, 1]
    fpr, tpr, _ = roc_curve(y[test_idx], y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Fold {i+1} AUC = {roc_auc:.3f}')

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve per Fold')
plt.legend()
plt.show()

#feature importnace across folds
importances = np.zeros(X.shape[1])
for train_idx, test_idx in cv.split(X, y):
    pipeline.fit(X[train_idx], y[train_idx])
    importances += pipeline.named_steps['randomforestclassifier'].feature_importances_

importances /= cv.get_n_splits()
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Average Feature Importances over CV folds")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [f"Feature {i}" for i in indices], rotation=90)
plt.tight_layout()
plt.show()
