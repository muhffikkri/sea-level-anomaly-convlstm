import numpy as np
import pandas as pd

def extract_north_java_coastline(dataset_2d, variable='sla'):
    """
    Mencari grid sel terakhir (laut) sebelum mengenai daratan (NaN)
    dari arah Utara ke Selatan.
    """
    latitudes = dataset_2d.latitude.values
    longitudes = dataset_2d.longitude.values
    data_array = dataset_2d[variable].values

    coastline_coords = []

    for lon_idx in range(2, len(longitudes)-2):
        for lat_idx in range(len(latitudes) - 2, -1, -1):
            val = data_array[lat_idx, lon_idx]
            if np.isnan(val): continue
            
            # Tetangga (Logic dari snippet Anda)
            l1, l2, l3 = data_array[lat_idx, lon_idx-1], data_array[lat_idx, lon_idx-2], data_array[lat_idx-1, lon_idx-2]
            r1, r2, r3 = data_array[lat_idx, lon_idx+1], data_array[lat_idx, lon_idx+2], data_array[lat_idx-1, lon_idx+2]
            b1, b2 = data_array[lat_idx-1, lon_idx], data_array[lat_idx-2, lon_idx]

            if ((np.isnan(l1) and np.isnan(l2) and np.isnan(l3)) or \
                (np.isnan(r1) and np.isnan(r2) and np.isnan(r3)) or \
                (np.isnan(b1) and np.isnan(b2))):
                
                if not np.isnan(data_array[lat_idx:, lon_idx]).sum() > 2:
                    coastline_coords.append({
                        'lon_idx': lon_idx,
                        'lat_idx': lat_idx,
                        'latitude': latitudes[lat_idx],
                        'longitude': longitudes[lon_idx]
                    })

    return pd.DataFrame(coastline_coords)