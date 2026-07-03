from data_integrity import check_files
from data_processing import run_extraction
from data_exploration import perform_eda
from pathlib import Path
from feature_engineering import prepare_convlstm_data, split_data
import joblib 
import numpy as np
import os

# --- KONFIGURASI ---
# Menggunakan Path akan otomatis menangani masalah slash
BASE_PATH = Path(r"E:\Smart Flood\sea-level-dataset-scraper-copernicus-main\data\data_2016_2025")
OUTPUT_DIR = Path(r"E:\Smart Flood\data\processed")
OUTPUT_FILE = OUTPUT_DIR / r"north_java_coast_2016_2025.parquet"
# main.py

# Pastikan folder 'processed' sudah ada, jika belum buat otomatis
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Langkah 1: Cek Integritas Data Mentah
    all_files = check_files(BASE_PATH)
    
    if all_files:
        # Langkah 2: Ekstraksi Data (Hanya jalankan jika file parquet belum ada atau ingin update)
        if not os.path.exists(OUTPUT_FILE):
            run_extraction(all_files, OUTPUT_FILE)
        else:
            print(f"\nℹ️ File {OUTPUT_FILE} sudah ada, melompati tahap ekstraksi.")

        # Langkah 3: EDA pada data hasil ekstraksi
        perform_eda(OUTPUT_FILE)

        # Langkah 4: Preprocessing untuk ConvLSTM
        WINDOW_SIZE = 30  
        LEAD_TIME = 1     
        
        print(f"\n=== Memulai Preprocessing (Lead Time: {LEAD_TIME} hari) ===")
        X, y, scaler = prepare_convlstm_data(OUTPUT_FILE, window_size=WINDOW_SIZE, lead_time=LEAD_TIME)
        
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        print(f"Shape Input (Samples, Time, H, W, C): {X_train.shape}")
        print(f"Data Training: {len(X_train)} sampel")
        print(f"Data Testing: {len(X_test)} sampel")
        
        # Simpan data siap pakai (Numpy Format)
        np.save("X_train.npy", X_train)
        np.save("y_train.npy", y_train)
        print("✅ Preprocessing selesai. Data siap untuk Training!")

if __name__ == "__main__":
    main()