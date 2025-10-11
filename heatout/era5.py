import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import project_id

import numpy as np
import pandas as pd
import ee
import requests
import datetime
import os
import rasterio

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project=project_id)

