import os
import ee
import numpy as np
import pandas as pd
import requests
from tif2df import tif_to_df
from config import *

def dowcon_day(dataset, aoi, date):
    day_start = ee.Date(date)
    day_end = day_start.advance(1, 'day')
    filtered_dataset = dataset.filterDate(day_start, day_end).select(bands)
    single_image = filtered_dataset.first()
    
    if single_image.getInfo() is None:
        print(f"No data available for {date}")
        return
    
    clipped_image = single_image.clip(aoi)
    
    year = day_start.format('YYYY').getInfo()
    tiff_dir = os.path.join('dataset/tiff', year)
    csv_dir = os.path.join('dataset/csv', year)
    os.makedirs(tiff_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    try:
        download_id = ee.data.getDownloadId({
            'image': clipped_image.toFloat(),
            'bands': bands,
            'region': aoi,
            'scale': 11132,
            'format': 'GEO_TIFF'
        })
        download_url = ee.data.makeDownloadUrl(download_id)
        
        date_string = day_start.format('YYYY-MM-DD').getInfo()
        tiff_file = os.path.join(tiff_dir, str(date) + '.tif')
        response = requests.get(download_url)
        if response.status_code == 200:
            with open(tiff_file, 'wb') as f:
                f.write(response.content)
            print(f"Download successful! TIFF saved as {tiff_file}")
            
            df, crs = tif_to_df(tiff_file)
            csv_file = os.path.join(csv_dir, str(date) + '.csv')
            df.to_csv(csv_file, index=False)
            print(f"CSV saved as {csv_file}")
            print(f"DataFrame shape: {df.shape}")

            df_stats = df.replace([np.inf, -np.inf], np.nan).describe()

        else:
            print(f"Download failed for {date_string} with status code {response.status_code}. "
                  "The dataset may be too large for direct download.")
    except ee.ee_exception.EEException as e:
        print(f"Error downloading {date_string}: {str(e)}")
