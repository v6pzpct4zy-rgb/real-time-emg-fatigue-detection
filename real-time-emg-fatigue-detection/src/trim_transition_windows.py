import numpy as np
import pandas as pd

# Dosyaları yükle
X = np.load("X_windows.npy")
y = np.load("y_labels.npy")
groups = np.load("groups_subject.npy")
meta = pd.read_csv("window_metadata.csv")

print("Orijinal şekiller:")
print("X:", X.shape)
print("y:", y.shape)
print("groups:", groups.shape)
print("meta:", meta.shape)

# Güvenlik kontrolü
assert len(X) == len(y) == len(groups) == len(meta), "Dosya boyutları eşleşmiyor!"

# Yeni tutma maskesi
keep_mask = np.ones(len(meta), dtype=bool)

# Her subject-trial-phase-label grubu için ayrı incele
group_cols = ["subject_id", "trial", "phase_id", "label"]

for keys, group in meta.groupby(group_cols):
    idx = group.index.values
    label = group["label"].iloc[0]

    n = len(group)

    # Çok küçük grupsa hiç oynama
    if n < 10:
        continue

    if label == "pre_fatigue":
        # son %20'yi at
        cut_start = int(n * 0.8)
        drop_idx = idx[cut_start:]
        keep_mask[drop_idx] = False

    elif label == "fatigue":
        # ilk %20'yi at
        cut_end = int(n * 0.2)
        drop_idx = idx[:cut_end]
        keep_mask[drop_idx] = False

    # rest olduğu gibi kalıyor

# Uygula
X_trim = X[keep_mask]
y_trim = y[keep_mask]
groups_trim = groups[keep_mask]
meta_trim = meta[keep_mask].reset_index(drop=True)

print("\nTrim sonrası şekiller:")
print("X_trim:", X_trim.shape)
print("y_trim:", y_trim.shape)
print("groups_trim:", groups_trim.shape)
print("meta_trim:", meta_trim.shape)

print("\nYeni label dağılımı:")
print(pd.Series(y_trim).value_counts())

print("\nBenzersiz subject sayısı:", len(np.unique(groups_trim)))
print("Subject ID'ler:", np.unique(groups_trim))

# Kaydet
np.save("X_windows_trimmed.npy", X_trim)
np.save("y_labels_trimmed.npy", y_trim)
np.save("groups_subject_trimmed.npy", groups_trim)
meta_trim.to_csv("window_metadata_trimmed.csv", index=False)

print("\nKaydedildi:")
print("- X_windows_trimmed.npy")
print("- y_labels_trimmed.npy")
print("- groups_subject_trimmed.npy")
print("- window_metadata_trimmed.csv")