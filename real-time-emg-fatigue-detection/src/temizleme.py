import pandas as pd

# birleşik dosyayı oku
df = pd.read_csv("combined_emg_with_subject.csv")

print("İlk label dağılımı:")
print(df["label"].value_counts(dropna=False))

# sadece istediğimiz label'ları tut
valid_labels = ["rest", "pre_fatigue", "fatigue", "recovery"]
df = df[df["label"].isin(valid_labels)].copy()

# eksik değer varsa temizle
df = df.dropna(subset=["subject_id", "emg_filtered", "label"])

# subject_id integer olsun
df["subject_id"] = df["subject_id"].astype(int)

# zaman varsa sırala
sort_cols = [col for col in ["subject_id", "trial", "phase_id", "time_s"] if col in df.columns]
df = df.sort_values(sort_cols).reset_index(drop=True)

print("\nTemizlenmiş label dağılımı:")
print(df["label"].value_counts())

print("\nKalan kişi sayısı:", df["subject_id"].nunique())
print("Toplam satır sayısı:", len(df))

df.to_csv("cleaned_emg.csv", index=False)
print("\nKaydedildi: cleaned_emg.csv")