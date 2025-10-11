import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
from dowcon import dowcon_day
import pandas as pd
import ee
import datetime

ee.Authenticate()
ee.Initialize(project=project_id)

aoi = ee.Geometry.Polygon(coordinates, proj='EPSG:4326', geodesic=False)

dataset = ee.ImageCollection(collection)

start_date = '2016-01-01'
end_date = '2025-09-30'

start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
delta = datetime.timedelta(days=1)
current_date = start

while current_date < end:
    date_str = current_date.strftime('%Y-%m-%d')
    print(f"Processing {date_str}...")
    dowcon_day(dataset, aoi, date_str)
    current_date += delta
