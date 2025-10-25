import numpy as np
import pandas as pd
from typing import Optional, Union
from numpy.typing import ArrayLike

def compute_penman_monteith_et0(
    temperature: ArrayLike,
    dewpoint_temperature: ArrayLike,
    u_wind_10m: ArrayLike,
    v_wind_10m: ArrayLike,
    surface_pressure: ArrayLike,
    net_solar_radiation: ArrayLike,
    net_thermal_radiation: ArrayLike,
    dates: Optional[ArrayLike] = None,
    output_path: Optional[str] = None
) -> np.ndarray:
    
    inputs = [
        np.asarray(temperature),
        np.asarray(dewpoint_temperature),
        np.asarray(u_wind_10m),
        np.asarray(v_wind_10m),
        np.asarray(surface_pressure),
        np.asarray(net_solar_radiation),
        np.asarray(net_thermal_radiation)
    ]

    lengths = [len(arr) for arr in inputs]
    if len(set(lengths)) > 1:
        raise ValueError("All input arrays must have the same length.")
    if dates is not None and len(dates) != lengths[0]:
        raise ValueError("Dates array must match the length of other inputs.")

    
