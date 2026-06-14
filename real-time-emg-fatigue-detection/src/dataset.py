import pandas as pd
import glob
import os

folder_path = r"C:\Users\Özge\Desktop\emg_dataset"   # kendi klasör yolunu yaz
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

all_dfs = []

for i, file in enumerate(sorted(csv_files), start=1):
    df = pd.read_csv(file)

    # kişiyi ekle
    df["subject_id"] = i

    # dosya adını da saklamak faydalı olur
    df["source_file"] = os.path.basename(file)

    all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)

print("Toplam dosya sayısı:", len(csv_files))
print("Toplam satır sayısı:", len(combined_df))
print("Kolonlar:", combined_df.columns.tolist())
print(combined_df[["subject_id", "source_file", "label"]].head())

combined_df.to_csv("combined_emg_with_subject.csv", index=False)
print("Kaydedildi: combined_emg_with_subject.csv")