import os
import numpy as np
import pandas as pd

path = "../data/processed/north_java_coast_2016_2025.parquet"
df_ = pd.read_parquet(path)
df = pd.read_parquet(path).loc[:"2021-12-31"]
print("Training",df.shape[0]/df_.shape[0])
print("Validasi",df_.loc["2022-01-01":"2023-12-31"].shape[0]/df_.shape[0])
print("Testing",df_.loc["2024-01-01":].shape[0]/df_.shape[0])

# flatten semua nilai
flat_values = df.values.flatten()

quantile = np.quantile(flat_values, [0.5, 0.75, 0.90, 0.95]).round(2)
# hitung quantile
print(quantile) 
print(sum(flat_values>=quantile[-1]))
print(sum((flat_values<quantile[-1])&(flat_values>=quantile[-3])))
print(sum(flat_values<quantile[-3]))

print((sum(flat_values>=quantile[-1])/len(flat_values)).round(2))
print((sum((flat_values<quantile[-1])&(flat_values>=quantile[-3]))/len(flat_values)).round(2))
print((sum(flat_values<quantile[-3])/len(flat_values)).round(2))