import pandas as pd
import numpy as np

# Temizlenmiş veri
df = pd.read_csv("cleaned_emg.csv")

window_size = 200
step_size = 100

X = []
y = []
groups = []
meta = []

# Hangi kolonlar varsa onları kullan
group_cols = [col for col in ["subject_id", "trial", "phase_id", "label"] if col in df.columns]

print("Group kolonları:", group_cols)

total_groups = 0
total_windows = 0

for keys, group in df.groupby(group_cols):
    # zaman sırasını koru
    if "time_s" in group.columns:
        group = group.sort_values("time_s").reset_index(drop=True)
    else:
        group = group.reset_index(drop=True)

    signal = group["emg_filtered"].values
    label = group["label"].iloc[0]
    subject_id = int(group["subject_id"].iloc[0])

    trial = int(group["trial"].iloc[0]) if "trial" in group.columns else -1
    phase_id = int(group["phase_id"].iloc[0]) if "phase_id" in group.columns else -1

    if len(signal) < window_size:
        continue

    total_groups += 1

    for start in range(0, len(signal) - window_size + 1, step_size):
        end = start + window_size
        window = signal[start:end]

        X.append(window)
        y.append(label)
        groups.append(subject_id)

        meta.append({
            "subject_id": subject_id,
            "trial": trial,
            "phase_id": phase_id,
            "label": label,
            "start_idx": start,
            "end_idx": end
        })

        total_windows += 1

X = np.array(X, dtype=np.float32)
y = np.array(y)
groups = np.array(groups)
meta_df = pd.DataFrame(meta)

print("\n===== WINDOWING SONUÇLARI =====")
print("Toplam grup sayısı:", total_groups)
print("Toplam pencere sayısı:", total_windows)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("groups shape:", groups.shape)

print("\nLabel bazlı pencere sayısı:")
print(pd.Series(y).value_counts())

print("\nBenzersiz subject sayısı:", len(np.unique(groups)))
print("Subject ID'ler:", np.unique(groups))

# Kaydet
np.save("X_windows.npy", X)
np.save("y_labels.npy", y)
np.save("groups_subject.npy", groups)
meta_df.to_csv("window_metadata.csv", index=False)

print("\nKaydedildi:")
print("- X_windows.npy")
print("- y_labels.npy")
print("- groups_subject.npy")
print("- window_metadata.csv")