import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
from tif2df import tif_to_df

import numpy as np
import pandas as pd
import ee
import requests
import datetime
import os
import rasterio

ee.Authenticate()
ee.Initialize(project=project_id)

aoi = ee.Geometry.Polygon(coordinates, proj='EPSG:4326', geodesic=False)

# print(aoi.getInfo())

start_date = '2016-01-01'
end_date = '2025-09-30'

dataset = ee.ImageCollection(collection)

