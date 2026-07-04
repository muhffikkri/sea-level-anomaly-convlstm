# =========================================================
# exploratory_data_analysis.py
# Exploratory Data Analysis Engine for Coastal SLA Tensor
# Fully Leakage-Safe: Percentiles calculated on Training Set Only
# =========================================================

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

from config import (
    FULL_GRID_TENSOR_FILE,
    TIMESTAMP_FILE,
    EXTREME_PERCENTILE,
    FIGURE_OUTPUT_DIR,
    METADATA_OUTPUT_DIR,
    TRAIN_END_DATE
)

# =========================================================
# MAIN EDA RECONSTRUCTION ENGINE
# =========================================================
def perform_eda():
    print("=" * 70)
    print("INITIALIZING LEAKAGE-SAFE EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # --- LOAD DATA ---
    print("[+] Loading SLA spatial tensor and timestamp logs...")
    sla_tensor = np.load(FULL_GRID_TENSOR_FILE)
    timestamps = pd.to_datetime(np.load(TIMESTAMP_FILE))
    print(f"[+] Loaded Tensor Shape : {sla_tensor.shape} (Time, H, W)")

    # --- COMPUTE REGIONAL SURFACE OCEAN STATISTICS ---
    print("[+] Computing regional statistics (ignoring landmass NaN values)...")
    regional_mean = np.nanmean(sla_tensor, axis=(1, 2))
    regional_std = np.nanstd(sla_tensor, axis=(1, 2))

    df_regional = pd.DataFrame({
        "sla_mean": regional_mean,
        "sla_std": regional_std
    }, index=timestamps)

    # =====================================================
    # CRITICAL SOLUTION: LEAKAGE-SAFE THRESHOLD CALCULATION
    # Percentile threshold must be locked on Training Set ONLY (2016-2021)
    # =====================================================
    print(f"[+] Isolating Training Set boundaries (Up to {TRAIN_END_DATE}) for P{EXTREME_PERCENTILE} computation...")
    train_mask = df_regional.index <= pd.to_datetime(TRAIN_END_DATE)
    train_regional_mean = df_regional.loc[train_mask, "sla_mean"].values
    
    # Menghitung batas ekstrem murni dari data latih
    extreme_threshold = np.percentile(train_regional_mean, EXTREME_PERCENTILE)
    print(f"✓ Success: Bounded Extreme Threshold Locked at : {extreme_threshold:.4f} meters")

    # Pencatatan kejadian ekstrem di seluruh data berdasarkan threshold data latih
    extreme_events = df_regional[df_regional["sla_mean"] >= extreme_threshold]

    # --- SAMPLE FINITE OCEAN PIXELS FOR DISTRIBUTION ANALYSIS ---
    flat_values = sla_tensor[np.isfinite(sla_tensor)]
    sample_size = min(300000, len(flat_values))
    np.random.seed(42) # Locked for strict reproducibility
    sampled_values = np.random.choice(flat_values, size=sample_size, replace=False)

    # --- PLOT INITIALIZATION ---
    sns.set_style("whitegrid")
    
    # =====================================================
    # FIGURE 1 — COMPREHENSIVE TEMPORAL ANALYSIS
    # =====================================================
    print("[+] Compiling Temporal Evaluation Panels (Figure 1)...")
    fig = plt.figure(figsize=(15, 12))

    # Panel 1: Long-term Trend with Moving Average
    plt.subplot(4, 1, 1)
    plt.plot(df_regional.index, df_regional["sla_mean"], alpha=0.3, color="gray", label="Daily Mean")
    plt.plot(df_regional.index, df_regional["sla_mean"].rolling(30).mean(), color="darkcyan", linewidth=2, label="30-Day Moving Average")
    plt.axhline(extreme_threshold, linestyle="--", color="black", linewidth=1.5, label=f"Training P{EXTREME_PERCENTILE} Boundary")
    plt.title("Regional Sea Level Anomaly Trend (2016-2025)", fontsize=10, fontweight="bold")
    plt.ylabel("SLA (m)")
    plt.legend(loc="upper right")

    # Panel 2: Monthly Variability (Boxplot to capture seasonal monsoon cycles)
    plt.subplot(4, 1, 2)
    df_regional["month"] = df_regional.index.month
    sns.boxplot(x="month", y="sla_mean", data=df_regional, palette="Blues")
    plt.title("Monthly SLA Seasonal Variability Profile", fontsize=10, fontweight="bold")
    plt.xlabel("Calendar Month")
    plt.ylabel("SLA (m)")

    # Panel 3: Probability Histogram
    plt.subplot(4, 1, 3)
    sns.histplot(sampled_values, bins=100, color="teal", kde=False, alpha=0.7)
    plt.axvline(extreme_threshold, linestyle="--", color="red", linewidth=1.5, label=f"Extreme Threshold ({extreme_threshold:.4f} m)")
    plt.title("SLA Empirical Histogram Distribution", fontsize=10, fontweight="bold")
    plt.xlabel("SLA (m)")
    plt.ylabel("Frequency")
    plt.legend()

    # Panel 4: Empirical Cumulative Distribution Function (ECDF)
    plt.subplot(4, 1, 4)
    sns.ecdfplot(sampled_values, color="darkblue", linewidth=2)
    plt.axvline(extreme_threshold, linestyle="--", color="red", linewidth=1.5, label=f"Extreme Boundary")
    plt.title("Empirical Cumulative Distribution Function (ECDF)", fontsize=10, fontweight="bold")
    plt.xlabel("SLA (m)")
    plt.ylabel("Probability")
    plt.legend()

    plt.tight_layout()
    figure_path = os.path.join(FIGURE_OUTPUT_DIR, "eda_temporal_analysis.png")
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure Panel 1 [Temporal Analysis]: {figure_path}")

    # =====================================================
    # FIGURE 2 — SPATIAL MEAN MAP WITH MASKED LANDMASS
    # =====================================================
    print("[+] Compiling Topological Spatial Mean Map (Figure 2)...")
    spatial_mean = np.nanmean(sla_tensor, axis=0)
    
    # Masking Daratan Jawa agar otomatis berwarna putih bersih di visualisasi
    spatial_mean_masked = np.ma.masked_where(np.isnan(spatial_mean), spatial_mean)
    cmap_spatial = plt.colormaps['viridis'].copy()
    cmap_spatial.set_bad('white')

    plt.figure(figsize=(7, 6))
    im = plt.imshow(spatial_mean_masked, origin="lower", cmap=cmap_spatial)
    plt.colorbar(im, label="Mean SLA (m)")
    plt.title("Spatial Mean Distribution Map\n(Landmass Preserved as Mask)", fontsize=10, fontweight="bold")
    plt.xlabel("Grid Longitude Index")
    plt.ylabel("Grid Latitude Index")

    spatial_path = os.path.join(FIGURE_OUTPUT_DIR, "spatial_mean_sla.png")
    plt.savefig(spatial_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure Panel 2 [Spatial Layout]: {spatial_path}")

    # =====================================================
    # FIGURE 3 — TIME-SERIES AUTOCORRELATION DECAY
    # =====================================================
    print("[+] Compiling Autocorrelation Decay Analysis (Figure 3)...")
    fig, ax = plt.subplots(figsize=(11, 4))
    plot_acf(regional_mean, lags=90, ax=ax, color="darkblue", vlines_color="skyblue")
    plt.title("Autocorrelation Function (ACF) of Regional SLA (90-Day Lag Window)", fontsize=10, fontweight="bold")
    plt.xlabel("Lag Days")
    plt.ylabel("Autocorrelation Force")

    acf_path = os.path.join(FIGURE_OUTPUT_DIR, "regional_sla_acf.png")
    plt.savefig(acf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure Panel 3 [Autocorrelation Profile]: {acf_path}")

    # --- COMPUTE COMPREHENSIVE RECAP STATISTICS ---
    autocorr_vals = {f"lag_{l}": float(pd.Series(regional_mean).autocorr(lag=l)) for l in [3, 7, 15, 30, 60]}

    summary_stats = {
        "mean_sla": float(np.nanmean(flat_values)),
        "std_sla": float(np.nanstd(flat_values)),
        "min_sla": float(np.nanmin(flat_values)),
        "max_sla": float(np.nanmax(flat_values)),
        "extreme_percentile": EXTREME_PERCENTILE,
        "extreme_threshold_meters": float(extreme_threshold),
        "total_historical_extreme_days": int(len(extreme_events)),
        "autocorrelation_decay": autocorr_vals
    }

    # --- SAVE ARTIFACT METADATA JSON ---
    summary_path = os.path.join(METADATA_OUTPUT_DIR, "eda_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_stats, f, indent=4)
    print(f"✓ Saved Empirical Metadata Log: {summary_path}")

    # --- CONSOLE SUMMARY DISPLAY ---
    print("\n" + "=" * 50)
    print("EMPIRICAL SUMMARY METRICS SUMMARY LOG")
    print("=" * 50)
    for key, value in summary_stats.items():
        if key != "autocorrelation_decay":
            print(f"{key:<30} : {value}")
    print("\nAutocorrelation Force Decay:")
    for lag, val in summary_stats["autocorrelation_decay"].items():
        print(f" - {lag:<10} : {val:.4f}")

    print("\n" + "=" * 50)
    print("TOP 5 HISTORICAL HYDRODYNAMIC SURGE EVENTS")
    print("=" * 50)
    print(extreme_events["sla_mean"].sort_values(ascending=False).head(5))
    print("=" * 70)

if __name__ == "__main__":
    perform_eda()