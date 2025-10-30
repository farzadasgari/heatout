import pandas as pd
import numpy as np

df_field = pd.read_csv("../dataset/heat.csv")
df_satel = pd.read_csv("../dataset/sentinel2.csv")
df_bands = pd.read_csv('../dataset/bsentinel.csv')

# Convert Date columns to datetime
df_field['Date'] = pd.to_datetime(df_field['Date'])
df_satel['Date'] = pd.to_datetime(df_satel['Date'], dayfirst=True)
df_bands['Date'] = pd.to_datetime(df_bands['Date'])

df = pd.merge(df_field, df_bands, on='Date', how='right')
df = pd.merge(df, df_satel, on='Date', how='right')

band_cols = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6',
             'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']

for col in band_cols:
    if col in df.columns:
        df[f'{col}'] = np.maximum(df[col] / 10000, 0)

# MNDWI = (Green - SWIR) / (Green + SWIR) = (B3 - B11) / (B3 + B11)
df['MNDWI'] = (df['B3'] - df['B11']) / (df['B3'] + df['B11'])

# GNDVI = (NIR - Green) / (NIR + Green) = (B8 - B3) / (B8 + B3)
df['GNDVI'] = (df['B8'] - df['B3']) / (df['B8'] + df['B3'])

# SDDI = Log(Green/Red) = Log(B3 / B4)
df['SDDI'] = np.log(df['B3'] / df['B4'])

# NDTI = (Red - Green) / (Red + Green) = (B4 - B3) / (B4 + B3)
df['NDTI'] = (df['B4'] - df['B3']) / (df['B4'] + df['B3'])

# BR = (Blue / Red) = B2 / B4
df['BR'] = df['B2'] / df['B4']

# NDWI = (Green - NIR) / (Green + NIR) = (B3 - B8) / (B3 + B8)
df['NDWI'] = (df['B3'] - df['B8']) / (df['B3'] + df['B8'])

# NDPI = (SWIR - Green) / (SWIR + Green) = (B11 - B3) / (B11 + B3)
df['NDPI'] = (df['B11'] - df['B3']) / (df['B11'] + df['B3'])

# NDCI = (RedEdge1 - Red) / (RedEdge1 + Red) = (B5 - B4) / (B5 + B4)
df['NDCI'] = (df['B5'] - df['B4']) / (df['B5'] + df['B4'])

# 2BDA (2-Band Difference Algorithm) for Chlorophyll-a Proxy: B5 - B4
df['2BDA_Chl'] = df['B5'] - df['B4']

# Red Edge / Red ratio for turbidity/sediment: B5 / B4
df['RR'] = df['B5'] / df['B4']

df.to_csv("../dataset/final.csv", index=False)
