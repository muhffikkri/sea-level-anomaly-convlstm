import xarray as xr
import pandas as pd
import glob
import os

def check_files(base_path):
    print("=== Memulai Pengecekan Integritas Data ===")
    file_pattern = os.path.join(base_path, "**", "*.nc")
    all_files = sorted(glob.glob(file_pattern, recursive=True))
    
    if not all_files:
        print("❌ Tidak ada file .nc ditemukan. Periksa path Anda.")
        return None

    # Menggunakan 'minimal' agar load metadata lebih cepat
    ds = xr.open_mfdataset(all_files, combine='nested', concat_dim='time', 
                           parallel=True, data_vars='minimal', coords='minimal')
    
    time_index = pd.to_datetime(ds.time.values)
    
    # 1. Analisis Metadata Perekaman
    print("\n--- Analisis Metadata Perekaman ---")
    # Mengecek apakah data ini rata-rata harian (Mean) atau snapshot
    cell_method = ds.sla.attrs.get('cell_methods', 'Tidak tersedia dalam metadata')
    print(f"Metode Sel (Cell Methods): {cell_method}")
    
    if "mean" in cell_method.lower():
        print("💡 Kesimpulan: Data ini adalah 'DAILY MEAN' (Rata-rata 24 jam).")
    else:
        print("💡 Kesimpulan: Data ini kemungkinan besar 'SNAPSHOT' (Titik waktu tertentu).")

    # 2. Ringkasan Jam Perekaman
    print("\n--- Ringkasan Waktu Perekaman (Timestamp) ---")
    df_time = pd.DataFrame({'hour': time_index.hour, 'minute': time_index.minute})
    time_summary = df_time.groupby(['hour', 'minute']).size().reset_index(name='count')
    
    for _, row in time_summary.iterrows():
        percentage = (row['count'] / len(time_index)) * 100
        print(f"⏰ Jam {int(row['hour']):02d}:{int(row['minute']):02d} UTC: {row['count']} file ({percentage:.1f}%)")

    # 3. Pengecekan Hari Bolong
    print("\n--- Mengecek Kelengkapan Tanggal ---")
    expected_range = pd.date_range(start=time_index.min(), end=time_index.max(), freq='D')
    missing_days = expected_range.difference(time_index)
    
    print(f"Rentang Data: {time_index.min().date()} s/d {time_index.max().date()}")
    if len(missing_days) == 0:
        print(f"✅ Integritas Terjamin: Tidak ada hari yang bolong (Total {len(time_index)} hari).")
    else:
        print(f"⚠️ Peringatan: Ada {len(missing_days)} hari yang hilang!")
    
    return all_files