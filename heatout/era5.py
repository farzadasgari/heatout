import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import project_id, coordinates

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
