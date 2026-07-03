import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(parquet_file):
    print("\n=== Memulai Eksplorasi Data Mendalam (Deep EDA) ===")
    df = pd.read_parquet(parquet_file)
    
    # --- 1. Analisis Tren Tahunan & Musiman ---
    # Kita buat rata-rata harian dari semua titik untuk melihat tren regional
    df_mean_regional = df.mean(axis=1).to_frame(name='sla_mean')
    
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Tren SLA 2016-2025 (Rolling Mean)
    plt.subplot(3, 1, 1)
    df_mean_regional['sla_mean'].rolling(window=30).mean().plot(color='red', label='30-Day Moving Average')
    df_mean_regional['sla_mean'].plot(alpha=0.3, color='gray', label='Daily Actual')
    plt.title("Tren Sea Level Anomaly (SLA) Regional - Pesisir Utara Jawa (2016-2025)")
    plt.ylabel("SLA (meter)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Boxplot Bulanan (Melihat Pola Musiman/Monsoon)
    plt.subplot(3, 1, 2)
    df_mean_regional['month'] = df_mean_regional.index.month
    sns.boxplot(x='month', y='sla_mean', data=df_mean_regional, palette='viridis')
    plt.title("Variasi Musiman SLA (Januari - Desember)")
    plt.xlabel("Bulan")
    plt.ylabel("SLA (meter)")

    # Plot 3: Distribusi Kumulatif (Melihat Probabilitas Ekstrim)
    plt.subplot(3, 1, 3)
    sns.ecdfplot(df.stack(), color='blue')
    plt.title("Cumulative Distribution Function (CDF) dari Seluruh Titik")
    plt.xlabel("SLA (meter)")
    plt.ylabel("Probability")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # --- 2. Deteksi Kejadian Ekstrim ---
    threshold = df_mean_regional['sla_mean'].mean() + (2 * df_mean_regional['sla_mean'].std())
    extreme_events = df_mean_regional[df_mean_regional['sla_mean'] > threshold]
    
    print(f"\n--- Analisis Kejadian Ekstrim ---")
    print(f"Rata-rata SLA Keseluruhan: {df_mean_regional['sla_mean'].mean():.4f} m")
    print(f"Ambang batas (Threshold) Anomali Tinggi (+2 std): {threshold:.4f} m")
    print(f"Jumlah hari dengan anomali tinggi: {len(extreme_events)} hari")
    
    print("\nTop 5 Hari dengan SLA Tertinggi (Potensi Rob Terparah):")
    print(extreme_events['sla_mean'].sort_values(ascending=False).head(5))