import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# DOSYA ADLARI
# ---------------------------
emg_file = r"C:\Users\Özge\Desktop\deneyler\yusufenes_emg.csv"
protocol_file = r"C:\Users\Özge\Desktop\deneyler\yusufenes.csv"
output_file = r"C:\Users\Özge\Desktop\deneyler\yusufenes_emg_labeled.csv"

# ---------------------------
# EMG DOSYASINI OKU
# ---------------------------
emg = pd.read_csv(emg_file)
emg["pc_timestamp"] = pd.to_datetime(emg["pc_timestamp"], errors="coerce")

# ---------------------------
# PROTOKOL DOSYASINI OKU
# ---------------------------
protocol = pd.read_csv(protocol_file)

protocol["start_time"] = pd.to_datetime(protocol["start_time"], utc=True, errors="coerce")
protocol["end_time"] = pd.to_datetime(protocol["end_time"], utc=True, errors="coerce")

protocol["start_time_local"] = (
    protocol["start_time"]
    .dt.tz_convert("Europe/Istanbul")
    .dt.tz_localize(None)
)

protocol["end_time_local"] = (
    protocol["end_time"]
    .dt.tz_convert("Europe/Istanbul")
    .dt.tz_localize(None)
)

# ---------------------------
# LABEL BAŞLATMA
# ---------------------------
emg["label"] = "unknown"
emg["trial"] = -1
emg["phase_name"] = "unknown"
emg["phase_id"] = -1

# ---------------------------
# ETİKETLEME
# ---------------------------
for _, row in protocol.iterrows():
    mask = (
        (emg["pc_timestamp"] >= row["start_time_local"]) &
        (emg["pc_timestamp"] <= row["end_time_local"])
    )

    emg.loc[mask, "label"] = row["label"]
    emg.loc[mask, "trial"] = row["trial"]
    emg.loc[mask, "phase_name"] = row["phase_name"]
    emg.loc[mask, "phase_id"] = row["phase_id"]

# ---------------------------
# KONTROL
# ---------------------------
print("Label dağılımı:")
print(emg["label"].value_counts(dropna=False))

emg.to_csv(output_file, index=False)
print(f"\nEtiketlenmiş dosya kaydedildi: {output_file}")

# ==========================================================
# POSTER İÇİN AKADEMİK EMG GRAFİĞİ
# ==========================================================

fig, ax = plt.subplots(figsize=(12, 4))

phase_colors = {
    "rest": "#d9f0d3",
    "pre_fatigue": "#fff2cc",
    "fatigue": "#f4cccc"
}

phase_names = {
    "rest": "Rest",
    "pre_fatigue": "Pre-fatigue",
    "fatigue": "Fatigue"
}

used_labels = set()

# Arka plan fazları
for _, row in protocol.iterrows():
    label = row["label"]

    if label not in phase_colors:
        continue

    phase_data = emg[
        (emg["pc_timestamp"] >= row["start_time_local"]) &
        (emg["pc_timestamp"] <= row["end_time_local"])
    ]

    if len(phase_data) == 0:
        continue

    start_t = phase_data["time_s"].iloc[0]
    end_t = phase_data["time_s"].iloc[-1]
    mid_t = (start_t + end_t) / 2

    legend_label = phase_names[label] if label not in used_labels else None
    used_labels.add(label)

    ax.axvspan(
        start_t,
        end_t,
        color=phase_colors[label],
        alpha=0.55,
        label=legend_label
    )

    # Faz ismini bölgenin içine yaz
    trial = row["trial"]
    phase_text = phase_names[label]

    if trial in [1, 2]:
        phase_text = f"{phase_names[label]} {int(trial)}"

    ax.text(
        mid_t,
        120,
        phase_text,
        ha="center",
        va="center",
        fontsize=9
    )
window = 50
emg["emg_smooth"] = emg["emg_filtered"].rolling(window, center=True).mean() 
# EMG sinyali
ax.plot(
    emg["time_s"],
    emg["emg_smooth"],
    color="red",
    linewidth=2,
    alpha=0.8,
    label="Smoothed EMG"
)

ax.set_xlabel("Zaman (s)")
ax.set_ylabel("EMG Genliği")
ax.set_title("Deney Protokolüne Göre Etiketlenmiş EMG Sinyali")

ax.set_ylim(-160, 160)
ax.grid(True, alpha=0.35)

ax.legend(loc="upper right", fontsize=8, frameon=True)

plt.tight_layout()
plt.savefig("emg_protocol_academic_style.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nPoster grafiği kaydedildi: emg_protocol_academic_style.png")