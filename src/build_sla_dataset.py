# =========================================================
# build_sla_dataset.py
# Build Full-Grid SLA Tensor Dataset (16x16 Preserved)
# Aligned with Unified ICICoS Geospatial Constraints
# =========================================================

import os
import json
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from config import (
    RAW_DATA_DIR,
    VARIABLE_NAME,
    MIN_LAT,
    MAX_LAT,
    MIN_LON,
    MAX_LON,
    FIGURE_OUTPUT_DIR,
    FULL_GRID_TENSOR_FILE,
    TIMESTAMP_FILE,
    COASTAL_POINT_FILE,
    METADATA_FILE,
    GRID_RESOLUTION,
    GRID_H,
    GRID_W
)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# =========================================================
# ADVANCED COASTLINE EXTRACTION (TRANSITION LOGIC)
# =========================================================
def extract_coastline_points(dataset_2d, variable=VARIABLE_NAME):
    """
    Extract exact North Java coastal interface nodes scanning from North to South.
    Filters out deep landward NaN boundaries to maintain a clean evaluation path.
    """
    latitudes = dataset_2d.latitude.values
    longitudes = dataset_2d.longitude.values
    data_array = dataset_2d[variable].values

    # Pastikan orientasi latitude bergerak dari Utara (nilai lebih besar) ke Selatan (lebih kecil)
    if latitudes[0] < latitudes[-1]:
        lat_direction = -1  # Urutan naik, maka loop mundur dari atas
    else:
        lat_direction = 1   # Urutan turun

    coastline_coords = []

    # Iterasi scanning kolom bujur (West to East)
    for lon_idx in range(len(longitudes)):
        # Pemindaian dari Utara (Laut) ke Selatan (Daratan Jawa) untuk menangkap bibir pantai pertama
        lat_indices = range(len(latitudes)) if lat_direction == 1 else range(len(latitudes)-1, -1, -1)
        
        found_coast_for_this_lon = False
        
        for lat_idx in lat_indices:
            val = data_array[lat_idx, lon_idx]
            
            # Lewati jika cell di laut lepas (bernilai finite angka SLA)
            if np.isfinite(val):
                continue
                
            # Jika menemukan NaN pertama kali setelah perairan, berarti ini batas intertidal / pantai
            if np.isnan(val) and not found_coast_for_this_lon:
                # Ambil cell valid terakhir di atasnya (tepat di air laut tepi pantai)
                actual_coast_lat = lat_idx - lat_direction
                
                if 0 <= actual_coast_lat < len(latitudes):
                    coast_val = data_array[actual_coast_lat, lon_idx]
                    if np.isfinite(coast_val):
                        coastline_coords.append({
                            "lon_idx": int(lon_idx),
                            "lat_idx": int(actual_coast_lat),
                            "latitude": float(latitudes[actual_coast_lat]),
                            "longitude": float(longitudes[lon_idx])
                        })
                        found_coast_for_this_lon = True
                        break

    coastline_df = pd.DataFrame(coastline_coords)
    coastline_df = coastline_df.drop_duplicates(subset=["lat_idx", "lon_idx"]).reset_index(drop=True)
    return coastline_df

# =========================================================
# GEOSPATIAL FILE ACQUISITION
# =========================================================
def get_all_netcdf_files():
    file_pattern = os.path.join(RAW_DATA_DIR, "**", "*.nc")
    nc_files = sorted(glob.glob(file_pattern, recursive=True))
    print(f"Total NetCDF Files Located : {len(nc_files)}")
    return nc_files

# =========================================================
# PUBLICATION-GRADE VISUALIZATIONS
# =========================================================
def visualize_sample_grid(sla_tensor, timestamps, sample_idx=0):
    plt.figure(figsize=(7, 6))
    # Masking daratan (NaN) agar otomatis berwarna putih bersih di visualisasi
    masked_tensor = np.ma.masked_where(np.isnan(sla_tensor[sample_idx]), sla_tensor[sample_idx])
    
    im = plt.imshow(masked_tensor, origin="lower", cmap="viridis")
    cbar = plt.colorbar(im, label="Sea Level Anomaly (m)")
    plt.title(f"SLA Subsampled Grid Landscape\nDate: {timestamps[sample_idx].strftime('%Y-%m-%d')}", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    save_path = os.path.join(FIGURE_OUTPUT_DIR, f"sample_grid_{RUN_TIMESTAMP}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure Landscape: {save_path}")

def visualize_coastal_points(sla_tensor, coastal_df, sample_idx=0):
    plt.figure(figsize=(7, 6))
    masked_tensor = np.ma.masked_where(np.isnan(sla_tensor[sample_idx]), sla_tensor[sample_idx])
    
    plt.imshow(masked_tensor, origin="lower", cmap="viridis")
    plt.scatter(coastal_df["lon_idx"], coastal_df["lat_idx"], c="red", edgecolors="white", s=35, label="Evaluation Nodes (13 Pts)")
    
    plt.colorbar(label="SLA (m)")
    plt.title("Spatial Distribution of Nearshore Evaluation Mask", fontsize=10, fontweight="bold")
    plt.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURE_OUTPUT_DIR, f"coastal_points_{RUN_TIMESTAMP}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Coastal Validation Map: {save_path}")

# =========================================================
# MAIN DATASET RECONSTRUCTION ENGINE
# =========================================================
def build_sla_dataset():
    print("=" * 70)
    print("INITIALIZING SPATIO-TEMPORAL SLA TENSOR EXTRACTION PIPELINE")
    print("=" * 70)

    nc_files = get_all_netcdf_files()
    if len(nc_files) == 0:
        raise FileNotFoundError(f"[-] Execution halted. Zero NetCDF assets detected in: {RAW_DATA_DIR}")

    tensor_list = []
    timestamps = []
    coastline_saved = False
    coastline_df = None

    # Pengondisian arah slice agar xarray tidak menghasilkan tensor kosong
    lat_slice = slice(MIN_LAT, MAX_LAT) if MIN_LAT < MAX_LAT else slice(MAX_LAT, MIN_LAT)
    lon_slice = slice(MIN_LON, MAX_LON) if MIN_LON < MAX_LON else slice(MAX_LON, MIN_LON)

    for file in tqdm(nc_files, desc="Parsing NetCDF Time-steps"):
        try:
            with xr.open_dataset(file) as ds:
                if VARIABLE_NAME not in ds:
                    continue

                # Pemotongan Spasial Menggunakan Jaminan Slice Searah
                subset = ds.sel(latitude=lat_slice, longitude=lon_slice)

                # Jaminan Penyelarasan Dimensi agar Selalu Tepat 16 x 16 Pixels
                if len(subset.latitude) != GRID_H or len(subset.longitude) != GRID_W:
                    # Jika dimensi selisih akibat pembulatan koordinat, paksa interpolasi ke resolusi target
                    subset = subset.interp(
                        latitude=np.linspace(MIN_LAT, MAX_LAT, GRID_H) if MIN_LAT < MAX_LAT else np.linspace(MAX_LAT, MIN_LAT, GRID_H),
                        longitude=np.linspace(MIN_LON, MAX_LON, GRID_W),
                        method="nearest"
                    )

                sla_grid = subset[VARIABLE_NAME].values
                if sla_grid.ndim == 3:
                    sla_grid = sla_grid[0]

                sla_grid = sla_grid.astype(np.float32)

                # Simpan Data Tensor & Waktu
                tensor_list.append(sla_grid)
                timestamps.append(pd.to_datetime(subset.time.values[0] if subset.time.values.ndim > 0 else subset.time.values))

                # Ekstraksi Masking Titik Pantai Secara Otomatis pada Iterasi Pertama
                if not coastline_saved:
                    print("\n[+] Extracting topological nearshore evaluation nodes...")
                    coastline_df = extract_coastline_points(subset.isel(time=0))
                    coastline_df.to_parquet(COASTAL_POINT_FILE, index=False)
                    print(f"[+] Total Coastal Cells Locked: {len(coastline_df)} Nodes (Target: 13 Pts)")
                    coastline_saved = True

        except Exception as e:
            print(f"\n[-] Error identified in processing file {os.path.basename(file)}: {e}")

    # Build Array Biner Akhir
    sla_tensor = np.stack(tensor_list, axis=0)
    timestamps = pd.to_datetime(timestamps)

    print("\n" + "-"*50)
    print(f"✓ Target SLA Shape Achieved : {sla_tensor.shape} (Time, H, W)")
    nan_ratio = np.isnan(sla_tensor).mean()
    print(f"✓ Macro Landmass NaN Ratio  : {nan_ratio:.4f}")
    print("-"*50)

    # Simpan Biner Numpy Secara Terpusat
    np.save(FULL_GRID_TENSOR_FILE, sla_tensor)
    np.save(TIMESTAMP_FILE, np.array(timestamps).astype("datetime64[D]"))

    # Dokumentasi Metadata JSON untuk Jaminan Reproduksibilitas Riset
    metadata = {
        "variable": VARIABLE_NAME,
        "tensor_shape": list(sla_tensor.shape),
        "grid_shape": {"rows": int(sla_tensor.shape[1]), "cols": int(sla_tensor.shape[2])},
        "spatial_extent": {"min_lon": MIN_LON, "max_lon": MAX_LON, "min_lat": MIN_LAT, "max_lat": MAX_LAT},
        "resolution_deg": GRID_RESOLUTION,
        "n_days": int(sla_tensor.shape[0]),
        "n_coastal_cells": int(len(coastline_df)),
        "start_date": str(timestamps.min().date()),
        "end_date": str(timestamps.max().date()),
        "tensor_dtype": str(sla_tensor.dtype),
        "nan_ratio": float(nan_ratio)
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

    # Jalankan Visualisasi Validasi Spasial
    print("\nGenerating structural diagnostic maps...")
    visualize_sample_grid(sla_tensor, timestamps)
    visualize_coastal_points(sla_tensor, coastline_df)

    print("\n" + "=" * 70)
    print("FULL GRID SLA TENSOR GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    build_sla_dataset()