# ============================================================
# WPT FEATURE EXTRACTION FOR EMG FATIGUE ANALYSIS
# Author: Özge
# Description:
#   This script extracts Wavelet Packet Transform (WPT) energy
#   features from windowed EMG signals and visualizes energy
#   distribution across fatigue states.
# ============================================================

import os
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt


# ============================================================
# 1) USER SETTINGS
# ============================================================

# CSV file path
CSV_PATH = "cleaned_emg.csv"

# Signal column used for analysis
SIGNAL_COL = "emg_filtered"

# Labels to analyze
VALID_LABELS = ["rest", "pre_fatigue", "fatigue"]

# Window parameters
WINDOW_SIZE = 200
STEP_SIZE = 100

# WPT parameters
WAVELET = "db4"
LEVEL = 4

# Sampling frequency
# IMPORTANT:
# Set this according to your real sampling frequency.
# If you are not fully sure, keep it as an assumed value and
# interpret frequency bands relatively.
FS = 1000  # Hz

# Output folder
OUTPUT_DIR = "wpt_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2) LOAD DATA
# ============================================================

print("Loading CSV file...")

df = pd.read_csv(CSV_PATH)

print("CSV loaded successfully.")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# ============================================================
# 3) BASIC CHECKS
# ============================================================

required_cols = [SIGNAL_COL, "label", "subject_id", "trial", "phase_id"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column not found: {col}")

# Keep only target labels
df = df[df["label"].isin(VALID_LABELS)].copy()

# Remove missing signal values
df = df.dropna(subset=[SIGNAL_COL])

print("\nFiltered dataset shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())


# ============================================================
# 4) FREQUENCY BAND LABELS
# ============================================================

def create_band_labels(fs, level):
    """
    Creates frequency band labels for WPT sub-bands.
    For level L, number of bands = 2^L.
    Frequency range is 0 to fs/2.
    """
    num_bands = 2 ** level
    nyquist = fs / 2
    band_width = nyquist / num_bands

    labels = []
    for i in range(num_bands):
        f_low = i * band_width
        f_high = (i + 1) * band_width
        labels.append(f"{f_low:.0f}-{f_high:.0f} Hz")

    return labels


band_labels = create_band_labels(FS, LEVEL)
print("\nFrequency bands:")
for i, b in enumerate(band_labels, start=1):
    print(f"Band {i}: {b}")


# ============================================================
# 5) WPT ENERGY FUNCTION
# ============================================================

def compute_wpt_energy(signal_window, wavelet="db4", level=4):
    """
    Computes WPT energy and normalized WPT energy
    for a single EMG window.
    """

    # Remove DC offset from window
    signal_window = signal_window - np.mean(signal_window)

    # Wavelet Packet Transform
    wp = pywt.WaveletPacket(
        data=signal_window,
        wavelet=wavelet,
        mode="symmetric",
        maxlevel=level
    )

    # Get nodes at selected level in frequency order
    nodes = wp.get_level(level, order="freq")

    energies = []

    for node in nodes:
        coeffs = node.data
        energy = np.sum(coeffs ** 2)
        energies.append(energy)

    energies = np.array(energies, dtype=np.float64)

    total_energy = np.sum(energies)

    if total_energy > 0:
        norm_energies = energies / total_energy
    else:
        norm_energies = np.zeros_like(energies)

    return energies, norm_energies


# ============================================================
# 6) WINDOWING + WPT FEATURE EXTRACTION
# ============================================================

print("\nStarting WPT feature extraction...")

rows = []

group_cols = ["subject_id", "trial", "phase_id", "label"]

for group_key, group_df in df.groupby(group_cols):

    subject_id, trial, phase_id, label = group_key

    # Sort by time if available
    if "time_s" in group_df.columns:
        group_df = group_df.sort_values("time_s")
    elif "time_sec" in group_df.columns:
        group_df = group_df.sort_values("time_sec")
    elif "sample" in group_df.columns:
        group_df = group_df.sort_values("sample")

    signal = group_df[SIGNAL_COL].values.astype(np.float32)

    if len(signal) < WINDOW_SIZE:
        continue

    for start in range(0, len(signal) - WINDOW_SIZE + 1, STEP_SIZE):

        end = start + WINDOW_SIZE
        window = signal[start:end]

        raw_energy, norm_energy = compute_wpt_energy(
            window,
            wavelet=WAVELET,
            level=LEVEL
        )

        row = {
            "subject_id": subject_id,
            "trial": trial,
            "phase_id": phase_id,
            "label": label,
            "start_idx": start,
            "end_idx": end
        }

        # Save raw WPT energy
        for i, e in enumerate(raw_energy):
            row[f"wpt_energy_b{i+1}"] = e

        # Save normalized WPT energy
        for i, e in enumerate(norm_energy):
            row[f"wpt_norm_energy_b{i+1}"] = e

        rows.append(row)

print("Feature extraction completed.")
print("Total windows:", len(rows))


# ============================================================
# 7) SAVE WPT FEATURES
# ============================================================

wpt_df = pd.DataFrame(rows)

output_csv = os.path.join(OUTPUT_DIR, "wpt_features.csv")
wpt_df.to_csv(output_csv, index=False)

print("\nWPT feature file saved:")
print(output_csv)

print("\nWPT feature dataframe shape:", wpt_df.shape)
print(wpt_df.head())


# ============================================================
# 8) CLASS-WISE MEAN WPT ENERGY
# ============================================================

norm_cols = [col for col in wpt_df.columns if col.startswith("wpt_norm_energy_b")]
raw_cols = [col for col in wpt_df.columns if col.startswith("wpt_energy_b")]

mean_norm_wpt = wpt_df.groupby("label")[norm_cols].mean()
mean_raw_wpt = wpt_df.groupby("label")[raw_cols].mean()

# Rename columns to frequency bands
mean_norm_wpt.columns = band_labels
mean_raw_wpt.columns = band_labels

mean_norm_csv = os.path.join(OUTPUT_DIR, "mean_normalized_wpt_energy_by_label.csv")
mean_raw_csv = os.path.join(OUTPUT_DIR, "mean_raw_wpt_energy_by_label.csv")

mean_norm_wpt.to_csv(mean_norm_csv)
mean_raw_wpt.to_csv(mean_raw_csv)

print("\nMean normalized WPT energy saved:")
print(mean_norm_csv)

print("\nMean raw WPT energy saved:")
print(mean_raw_csv)


# ============================================================
# 9) PLOT NORMALIZED WPT ENERGY HEATMAP
# ============================================================

plt.figure(figsize=(16, 4))

# Desired label order
label_order = ["rest", "pre_fatigue", "fatigue"]
mean_norm_plot = mean_norm_wpt.loc[label_order]

plt.imshow(mean_norm_plot.values, aspect="auto", cmap="YlOrRd")

plt.colorbar(label="Normalized Energy")

plt.xticks(
    ticks=np.arange(len(band_labels)),
    labels=band_labels,
    rotation=45,
    ha="right"
)

plt.yticks(
    ticks=np.arange(len(label_order)),
    labels=["Rest", "Pre-Fatigue", "Fatigue"]
)

plt.title("Normalized WPT Energy Heatmap – Fatigue State Discrimination")
plt.xlabel("Frequency Sub-band (Hz)")
plt.ylabel("Fatigue State")

plt.tight_layout()

heatmap_path = os.path.join(OUTPUT_DIR, "wpt_energy_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.show()

print("\nHeatmap saved:")
print(heatmap_path)


# ============================================================
# 10) PLOT MEAN RAW WPT ENERGY BAR CHART
# ============================================================

mean_raw_plot = mean_raw_wpt.loc[label_order]

x = np.arange(len(band_labels))
width = 0.25

plt.figure(figsize=(18, 6))

plt.bar(
    x - width,
    mean_raw_plot.loc["rest"].values,
    width,
    label="Rest"
)

plt.bar(
    x,
    mean_raw_plot.loc["pre_fatigue"].values,
    width,
    label="Pre-Fatigue"
)

plt.bar(
    x + width,
    mean_raw_plot.loc["fatigue"].values,
    width,
    label="Fatigue"
)

plt.xticks(
    x,
    band_labels,
    rotation=45,
    ha="right"
)

plt.xlabel("Frequency Sub-band (Hz)")
plt.ylabel("Mean WPT Energy")
plt.title("Wavelet Packet Transform – Mean Energy per Sub-band Across Fatigue States")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

bar_path = os.path.join(OUTPUT_DIR, "wpt_band_energy.png")
plt.savefig(bar_path, dpi=300)
plt.show()

print("\nBar chart saved:")
print(bar_path)


# ============================================================
# 11) OPTIONAL: PLOT NORMALIZED BAR CHART
# ============================================================

plt.figure(figsize=(18, 6))

plt.bar(
    x - width,
    mean_norm_plot.loc["rest"].values,
    width,
    label="Rest"
)

plt.bar(
    x,
    mean_norm_plot.loc["pre_fatigue"].values,
    width,
    label="Pre-Fatigue"
)

plt.bar(
    x + width,
    mean_norm_plot.loc["fatigue"].values,
    width,
    label="Fatigue"
)

plt.xticks(
    x,
    band_labels,
    rotation=45,
    ha="right"
)

plt.xlabel("Frequency Sub-band (Hz)")
plt.ylabel("Mean Normalized WPT Energy")
plt.title("Normalized WPT Energy per Sub-band Across Fatigue States")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

norm_bar_path = os.path.join(OUTPUT_DIR, "wpt_normalized_band_energy.png")
plt.savefig(norm_bar_path, dpi=300)
plt.show()

print("\nNormalized bar chart saved:")
print(norm_bar_path)


# ============================================================
# 12) SUMMARY
# ============================================================

print("\n========== SUMMARY ==========")
print("Input file:", CSV_PATH)
print("Signal column:", SIGNAL_COL)
print("Window size:", WINDOW_SIZE)
print("Step size:", STEP_SIZE)
print("Wavelet:", WAVELET)
print("WPT level:", LEVEL)
print("Number of sub-bands:", 2 ** LEVEL)
print("Sampling frequency used for band labels:", FS, "Hz")
print("Total WPT windows:", len(wpt_df))
print("Output folder:", OUTPUT_DIR)
print("=============================")