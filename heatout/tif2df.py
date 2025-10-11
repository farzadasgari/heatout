import numpy as np
import pandas as pd
import rasterio
from config import *

def tif_to_df(filepath):
    with rasterio.open(filepath) as src:
        transform = src.transform
        crs = src.crs
        data = src.read()
        nrows, ncols = data.shape[1], data.shape[2]
        
        x = np.arange(ncols) * transform.a + transform.c
        y = np.arange(nrows) * transform.e + transform.f
        xv, yv = np.meshgrid(x, y)
        
        df_dict = {
            'longitude': xv.flatten(),
            'latitude': yv.flatten()
        }
        
        for i, band_name in enumerate(bands):
            df_dict[band_name] = data[i].flatten()
        
        df = pd.DataFrame(df_dict)

        for band in bands:
            invalid_count = np.sum(~np.isfinite(df[band]))
        return df, crs
