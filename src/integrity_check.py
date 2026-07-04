# =========================================================
# integrity_check.py
# Dataset Integrity & Consistency Validation Engine
# Fully Synchronized with Unified ICICoS Subsampling System
# =========================================================

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

from config import (
    RAW_DATA_DIR,
    NETCDF_EXTENSION,
    VARIABLE_NAME,
    EXTREME_PERCENTILE,
    MIN_LAT,
    MAX_LAT,
    MIN_LON,
    MAX_LON,
    GRID_RESOLUTION,
    GRID_H,
    GRID_W
)

# =========================================================
# MAIN INTEGRITY VALIDATION ENGINE
# =========================================================
def check_dataset_integrity():
    print("=" * 70)
    print("INITIALIZING DATASET INTEGRITY & GEOSPATIAL CONSISTENCY CHECK")
    print("=" * 70)

    # --- CORRECTION: Safe String-Based Path Compilation ---
    file_pattern = os.path.join(RAW_DATA_DIR, "**", f"*{NETCDF_EXTENSION}")
    all_files = sorted(glob.glob(file_pattern, recursive=True))

    if len(all_files) == 0:
        raise FileNotFoundError(f"[-] Execution Halted. Zero {NETCDF_EXTENSION} assets located in: {RAW_DATA_DIR}")

    print(f"[+] Total NetCDF file nodes discovered: {len(all_files)}")
    print("[+] Validating coordinates and slicing parameters...")

    # Membatasi arah irisan geografis agar xarray tidak menghasilkan subset kosong
    lat_slice = slice(MIN_LAT, MAX_LAT) if MIN_LAT < MAX_LAT else slice(MAX_LAT, MIN_LAT)
    lon_slice = slice(MIN_LON, MAX_LON) if MIN_LON < MAX_LON else slice(MAX_LON, MIN_LON)

    # Memuat metadata berkas secara paralel menggunakan Dask Lazy Processing
    ds = xr.open_mfdataset(
        all_files,
        combine="nested",
        concat_dim="time",
        parallel=True,
        data_vars="minimal",
        coords="minimal",
        compat="override"
    )

    # Melakukan pemotongan spasial langsung pada dataset induk untuk validasi dimensi
    ds_subset = ds.sel(latitude=lat_slice, longitude=lon_slice)

    print("\n" + "=" * 50)
    print("1. DIMENSION & SUBSAMPLING RESOLUTION PROFILE")
    print("=" * 50)
    print(f"Target Variable Name    : {VARIABLE_NAME}")
    print(f"Total Temporal Samples  : {ds_subset.sizes['time']} Days")
    print(f"Subsampled Latitude     : {ds_subset.sizes['latitude']} Pixels (Target Config: {GRID_H})")
    print(f"Subsampled Longitude    : {ds_subset.sizes['longitude']} Pixels (Target Config: {GRID_W})")

    lat_res = np.abs(np.diff(ds_subset.latitude.values)).mean()
    lon_res = np.abs(np.diff(ds_subset.longitude.values)).mean()
    print(f"Empirical Resolution   : Lat: {lat_res:.4f}° | Lon: {lon_res:.4f}° (Config: {GRID_RESOLUTION}°)")

    print("\n" + "=" * 50)
    print("2. TEMPORAL CHRONOLOGICAL CONSISTENCY CHECK")
    print("=" * 50)
    time_index = pd.to_datetime(ds_subset.time.values)
    print(f"Temporal Boundaries    : {time_index.min().date()} ---> {time_index.max().date()}")
    
    # Deteksi Urutan Kronologis (Sangat Penting untuk Mencegah Kebocoran Deret Waktu)
    if time_index.is_monotonic_increasing:
        print("✓ Success: Timestamps are perfectly ordered chronologically.")
    else:
        print("⚠ Warning: Timestamps are NOT chronologically sorted! Check folder names.")

    # Deteksi Jeda Interval Waktu
    time_diffs = np.diff(time_index.values).astype("timedelta64[D]")
    unique_diffs = np.unique(time_diffs)
    print(f"Identified intervals   : {[str(d) for d in unique_diffs]}")

    # Pemeriksaan Kebocoran Tanggal (Missing Dates)
    expected_dates = pd.date_range(start=time_index.min(), end=time_index.max(), freq="D")
    missing_dates = expected_dates.difference(time_index)
    if len(missing_dates) == 0:
        print("✓ Success: No missing calendar dates detected.")
    else:
        print(f"⚠ Warning: Detected {len(missing_dates)} missing dates in the sequence!")
        for date in missing_dates[:5]:
            print(f"   - Missing: {date.date()}")

    print("\n" + "=" * 50)
    print("3. LANDMASS MASK & STABILITY ANALYSIS")
    print("=" * 50)
    # Memeriksa kestabilan masker daratan (NaN) dari awal hingga akhir rentang waktu
    mask_start = np.isnan(ds_subset[VARIABLE_NAME].isel(time=0).values)
    mask_end = np.isnan(ds_subset[VARIABLE_NAME].isel(time=-1).values)
    
    if np.array_equal(mask_start, mask_end):
        print("✓ Success: Land/Ocean grid mask remains perfectly static over time.")
    else:
        print("⚠ Critical: Landmask fluctuates! Missing values dynamically changing.")

    print("\n" + "=" * 50)
    print("4. VALUE RANGE & EXTREMESurge DISTRIBUTION")
    print("=" * 50)
    
    # --- OPTIMIZATION: Single Memory Fetch to Prevent RAM Blow-up ---
    print("Computing global spatial stats (Fetching matrix logs into memory)...")
    variable_ds = ds_subset[VARIABLE_NAME]
    
    # Menghitung seluruh statistik dasar dalam satu ketukan grafik Dask
    stats_bundle = xr.Tuple([
        variable_ds.min(),
        variable_ds.max(),
        variable_ds.mean(),
        variable_ds.std(),
        variable_ds.quantile(EXTREME_PERCENTILE / 100)
    ]).compute()

    data_min, data_max, data_mean, data_std, extreme_threshold = [float(val) for val in stats_bundle]

    print(f"Absolute Minimum SLA   : {data_min:.4f} m")
    print(f"Absolute Maximum SLA   : {data_max:.4f} m")
    print(f"Baseline Mean SLA      : {data_mean:.4f} m")
    print(f"Standard Deviation     : {data_std:.4f} m")
    print(f"Target Extreme Threshold ({EXTREME_PERCENTILE}th Pctl) : {extreme_threshold:.4f} m")

    print("\n" + "=" * 50)
    print("5. TIME-SERIES AUTOCORRELATION DECAY PROFILE")
    print("=" * 50)
    # Menghitung autokorelasi wilayah makro untuk pemodelan lag temporal
    regional_mean = variable_ds.mean(dim=["latitude", "longitude"]).to_series()
    print(f"Lag-1  Autocorrelation (Daily Persistence Force) : {regional_mean.autocorr(lag=1):.4f}")
    print(f"Lag-7  Autocorrelation (Weekly Cycle Decay)     : {regional_mean.autocorr(lag=7):.4f}")
    print(f"Lag-30 Autocorrelation (Monthly Tidal Boundary) : {regional_mean.autocorr(lag=30):.4f}")

    print("\n" + "=" * 70)
    print("✓ DATASET INTEGRITY VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    ds.close()
    return all_files

if __name__ == "__main__":
    check_dataset_integrity()