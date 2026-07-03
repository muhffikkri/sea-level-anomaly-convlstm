import os
import json
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# ==========================================
# CONFIG
# ==========================================

ROOT_DIR = r"E:\Smart Flood\sea-level-dataset-scraper-copernicus-main\data\data_2016_2025"

MIN_LON = 109.5
MAX_LON = 111.5

MIN_LAT = -8.0
MAX_LAT = -6.0

VARIABLE = "sla"

# ==========================================
# COASTLINE FUNCTION
# ==========================================

def extract_north_java_coastline(dataset_2d, variable='sla'):

    latitudes = dataset_2d.latitude.values
    longitudes = dataset_2d.longitude.values
    data_array = dataset_2d[variable].values

    coastline_coords = []

    for lon_idx in range(2, len(longitudes)-2):

        for lat_idx in range(len(latitudes)-2, -1, -1):

            val = data_array[lat_idx, lon_idx]

            l1 = data_array[lat_idx, lon_idx-1]
            l2 = data_array[lat_idx, lon_idx-2]
            l3 = data_array[lat_idx-1, lon_idx-2]

            r1 = data_array[lat_idx, lon_idx+1]
            r2 = data_array[lat_idx, lon_idx+2]
            r3 = data_array[lat_idx-1, lon_idx+2]

            b1 = data_array[lat_idx-1, lon_idx]
            b2 = data_array[lat_idx-2, lon_idx]

            if (
                not np.isnan(val)
                and (
                    (np.isnan(l1) and np.isnan(l2) and np.isnan(l3))
                    or
                    (np.isnan(r1) and np.isnan(r2) and np.isnan(r3))
                    or
                    (np.isnan(b1) and np.isnan(b2))
                )
            ):

                if not np.isnan(
                    data_array[lat_idx:, lon_idx]
                ).sum() > 2:

                    coastline_coords.append({
                        "lon_idx": lon_idx,
                        "lat_idx": lat_idx,
                        "latitude": latitudes[lat_idx],
                        "longitude": longitudes[lon_idx]
                    })

    return pd.DataFrame(coastline_coords)

# ==========================================
# FIND ALL NETCDF FILES
# ==========================================

nc_files = []

for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.endswith(".nc"):
            nc_files.append(os.path.join(root, file))

nc_files = sorted(nc_files)

print(f"Total files: {len(nc_files)}")

# ==========================================
# PROCESS
# ==========================================

tensor_list = []
timestamps = []

coastline_saved = False

for file in tqdm(nc_files):

    try:

        ds = xr.open_dataset(file)

        subset = ds.sel(
            longitude=slice(MIN_LON, MAX_LON),
            latitude=slice(MIN_LAT, MAX_LAT)
        )

        sla_grid = subset[VARIABLE].values

        # Handle shape
        if sla_grid.ndim == 3:
            sla_grid = sla_grid[0]

        tensor_list.append(
            sla_grid.astype(np.float32)
        )

        # timestamp
        timestamps.append(
            pd.to_datetime(
                subset.time.values[0]
            )
        )

        # coastline extraction sekali saja
        if not coastline_saved:

            coastline_df = extract_north_java_coastline(
                subset.isel(time=0),
                variable=VARIABLE
            )

            coastline_df.to_parquet(
                "coastal_points.parquet",
                index=False
            )

            coastline_saved = True

        ds.close()

    except Exception as e:

        print(file)
        print(e)

# ==========================================
# BUILD TENSOR
# ==========================================

sla_tensor = np.stack(
    tensor_list,
    axis=0
)

print("Tensor shape:")
print(sla_tensor.shape)

# ==========================================
# SAVE
# ==========================================

np.save(
    "sla_tensor.npy",
    sla_tensor
)

np.save(
    "timestamps.npy",
    np.array(timestamps).astype("datetime64[D]")
)

# ==========================================
# METADATA
# ==========================================

metadata = {

    "variable": VARIABLE,

    "min_lon": MIN_LON,
    "max_lon": MAX_LON,

    "min_lat": MIN_LAT,
    "max_lat": MAX_LAT,

    "grid_rows": int(sla_tensor.shape[1]),
    "grid_cols": int(sla_tensor.shape[2]),

    "resolution_deg": 0.125,

    "n_days": int(sla_tensor.shape[0]),

    "coastal_cells": int(
        len(coastline_df)
    )
}

with open(
    "metadata.json",
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )

print("Finished.")