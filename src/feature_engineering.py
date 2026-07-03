import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def prepare_convlstm_data(parquet_file, window_size=30, lead_time=1):
    """
    Menyiapkan data sekuensial gridded untuk ConvLSTM sesuai draf penelitian
    Format output: (Samples, Time, Height, Width, Channels)
    """
    df = pd.read_parquet(parquet_file)
    
    # 1. Normalisasi Data ke rentang 0-1
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_data = scaler.fit_transform(df.values)
    
    # 2. Reshape ke Grid Spasial
    # Berdasarkan 82 titik yang ditemukan, kita bentuk grid 9x10 (total 90 cell)
    # num_samples = len(scaled_data)
    num_samples = len(df.values)
    height, width = 9, 10 
    
    # Tambahkan padding nol agar pas menjadi 90 titik (9x10)
    padding = np.zeros((num_samples, (height * width) - df.shape[1]))
    grid_data = np.hstack([scaled_data, padding])
    grid_data = grid_data.reshape(num_samples, height, width, 1) # 1 Channel: SLA

    # 3. Implementasi Sliding Window 
    X, y = [], []
    for i in range(num_samples - window_size - lead_time + 1):
        # Input: Urutan gambar SLA selama 'window_size' hari
        X.append(grid_data[i : i + window_size])
        
        # Target: Nilai SLA pada t + lead_time (1, 3, atau 7 hari)
        # Mengambil rata-rata regional sebagai target regresi
        y.append(grid_data[i + window_size + lead_time - 1].mean()) 

    return np.array(X), np.array(y), scaler

def split_data(X, y, train_ratio=0.8):
    """Pemisahan data secara kronologis untuk data deret waktu."""
    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]