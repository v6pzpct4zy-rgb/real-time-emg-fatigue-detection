import numpy as np
import pandas as pd

# yükle
X = np.load("X_windows.npy")
y = np.load("y_labels.npy")
groups = np.load("groups_subject.npy")

def mean_frequency(window):
    fft_vals = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(len(window), d=1/1000)

    return np.sum(freqs * fft_vals) / np.sum(fft_vals)

def median_frequency(window):
    fft_vals = np.abs(np.fft.rfft(window))
    cumulative = np.cumsum(fft_vals)
    total = cumulative[-1]

    return np.where(cumulative >= total/2)[0][0]
# feature hesaplama fonksiyonları
def extract_features(window):
    features = {}

    # ÖNEMLİ: DC offset kaldır
    window = window - np.mean(window)
    features["MNF"] = mean_frequency(window)
    features["RMS"] = np.sqrt(np.mean(window**2))
    features["MAV"] = np.mean(np.abs(window))
    features["VAR"] = np.var(window)
    features["WL"] = np.sum(np.abs(np.diff(window)))
    
    threshold = 0.01
    zc = np.sum((window[:-1] * window[1:] < 0) & 
                (np.abs(window[:-1] - window[1:]) > threshold))
    features["ZC"] = zc

    return features

# tüm veriye uygula
feature_list = []

for i in range(len(X)):
    f = extract_features(X[i])
    feature_list.append(f)

df_features = pd.DataFrame(feature_list)

# label ve subject ekle
df_features["label"] = y
df_features["subject_id"] = groups

print("\n===== FEATURE TABLOSU =====")
print(df_features.head())

print("\nLabel dağılımı:")
print(df_features["label"].value_counts())

print("\nFeature boyutu:", df_features.shape)

df_features.to_csv("features.csv", index=False)

print("\nKaydedildi: features.csv")