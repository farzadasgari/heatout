import numpy as np
import pandas as pd
from typing import Optional, Union, Any
from numpy.typing import ArrayLike
from pint import UnitRegistry

ureg = UnitRegistry()
ureg.setup_matplotlib()
Quantity = ureg.Quantity

def compute_penman_monteith_et0(
    temperature: ArrayLike,
    dewpoint_temperature: ArrayLike,
    u_wind_10m: ArrayLike,
    v_wind_10m: ArrayLike,
    surface_pressure: ArrayLike,
    net_solar_radiation: ArrayLike,
    net_thermal_radiation: ArrayLike,
    dates: Optional[ArrayLike] = None,
    output_path: Optional[str] = None,
    units: Optional[str] = None,
) -> np.ndarray:

    def to_quantity(value, default_unit):
        if isinstance(value, Quantity):
            return value
        return value * default_unit
    
    T = to_quantity(temperature, ureg.kelvin)
    Td = to_quantity(dewpoint_temperature, ureg.kelvin)
    u10 = to_quantity(u_wind_10m, ureg.meter / ureg.second)
    v10 = to_quantity(v_wind_10m, ureg.meter / ureg.second)
    P = to_quantity(surface_pressure, ureg.pascal)
    Rn_sw = to_quantity(net_solar_radiation, ureg.watt / ureg.meter**2)
    Rn_lw = to_quantity(net_thermal_radiation, ureg.watt / ureg.meter**2)

    T_C = T.to(ureg.degC)
    Td_C = Td.to(ureg.degC)

    lengths = [len(np.asarray(q.magnitude)) for q in [T, Td, u10, v10, P, Rn_sw, Rn_lw]]
    if len(set(lengths)) > 1:
        raise ValueError("All input arrays must have the same length.")
    if dates is not None and len(dates) != lengths[0]:
        raise ValueError("Dates array must match the length of other inputs.")
    
    es = (0.6108 * np.exp(17.27 * T_C.magnitude / (T_C.magnitude + 237.3))) * ureg.kPa
    ea = (0.6108 * np.exp(17.27 * Td_C.magnitude / (Td_C.magnitude + 237.3))) * ureg.kPa
    es_ea = es - ea
    
