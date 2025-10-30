import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
import ee
import requests
import datetime
import rasterio
import numpy as np
import pandas as pd

ee.Authenticate()
ee.Initialize(project=project_id)

start_date = '2014-01-01'
end_date = '2025-01-01'

aoi = ee.Geometry.Polygon(coordinates, proj='EPSG:4326', geodesic=False)

dataset = ee.ImageCollection(bcollection)

filtered = dataset.filterDate(start_date, end_date) \
                  .filterBounds(aoi) \
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

selected = filtered.select(bbands)

def mosaic_by_date(col):
    dates = col.toList(col.size()).map(lambda img: ee.Image(img).date().format('YYYY-MM-dd')).distinct()
    
    def create_mosaic(date_str):
        date = ee.Date(date_str)
        day_imgs = col.filterDate(date, date.advance(1, 'day')).mosaic()
        return day_imgs.set('Date', date.format('YYYY-MM-dd'))
    
    return ee.ImageCollection(dates.map(create_mosaic))

def compute_means(image):
    means = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10,
        bestEffort=True
    )
    date = image.get('Date')
    return ee.Feature(None, means).set('Date', date)

mosaicked_collection = mosaic_by_date(selected)
means_collection = ee.FeatureCollection(mosaicked_collection.map(compute_means))

download_url = means_collection.getDownloadURL('CSV')

temp_csv_file = 'temp.csv'

response = requests.get(download_url)
with open(temp_csv_file, 'wb') as f:
    f.write(response.content)

df = pd.read_csv(temp_csv_file)

band_order = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']

available_bands = [col for col in df.columns if col in band_order]
sorted_bands = sorted(available_bands, key=lambda x: band_order.index(x) if x in band_order else len(band_order))

columns_to_remove = ['system:index', '.geo']
columns_to_keep = ['Date'] + sorted_bands

other_columns = [col for col in df.columns 
                if col not in columns_to_remove 
                and col not in columns_to_keep]

columns_to_keep.extend(other_columns)

df_clean = df[columns_to_keep]

csv_file = '../dataset/bsentinel.csv'
df_clean.to_csv(csv_file, index=False)

os.remove(temp_csv_file)

print(f"Download and processing successful! Cleaned CSV saved")
