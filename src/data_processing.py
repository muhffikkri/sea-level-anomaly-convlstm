import xarray as xr
import pandas as pd
from extract_coastline import extract_north_java_coastline

def run_extraction(all_files, output_name="north_java_coast_2016_2025.parquet"):
    print("\n=== Memulai Ekstraksi Data Pesisir ===")
    
    # Load Dataset
    ds = xr.open_mfdataset(all_files, combine='nested', concat_dim='time', 
                           chunks={'time': 100}, parallel=True)

    # Koordinat Sesuai Permintaan
    min_lat, max_lat = -8.5, -5.5 
    min_lon, max_lon = 105, 115
    
    # Slicing Area
    ds_area = ds.sel(latitude=slice(min_lat, max_lat), longitude=slice(min_lon, max_lon))
    
    # Ambil snapshot untuk deteksi garis pantai (time index 0)
    print("Mendeteksi garis pantai...")
    ds_snapshot = ds_area.isel(time=0).load()
    df_coastline = extract_north_java_coastline(ds_snapshot)
    
    print(f"Ditemukan {len(df_coastline)} titik pesisir.")

    # Ekstraksi Time Series Massal
    points_lat = xr.DataArray(df_coastline['latitude'].values, dims="points")
    points_lon = xr.DataArray(df_coastline['longitude'].values, dims="points")

    print("Mengekstraksi nilai SLA (ini memakan waktu)...")
    coastal_data = ds_area['sla'].sel(latitude=points_lat, longitude=points_lon, method='nearest')
    
    # Simpan
    df_final = coastal_data.to_pandas()
    df_final.to_parquet(output_name)
    print(f"✅ Selesai! Data disimpan di: {output_name}")
    
    return df_final