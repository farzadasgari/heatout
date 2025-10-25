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

    delta = (4098 * es / ((T_C.magnitude + 237.3) ** 2))
    gamma = 0.000665 * P.to(ureg.kPa)
    
    ln_term = np.log(67.8 * 10 - 5.42)
    u2 = u10_total * (4.87 / ln_term)

    Rn_sw_daily = Rn_sw * (3600 * 24)
    Rn_lw_daily = Rn_lw * (3600 * 24)
    Rn_daily = Rn_sw_daily + Rn_lw_daily

    G = 0 * ureg.megajoule / ureg.meter**2 / ureg.day

    term1 = delta * (Rn_daily - G)
    term2 = gamma * (900 / T.to(ureg.kelvin).magnitude) * u2 * es_ea
    
    numerator = term1 + term2
    denominator = delta + gamma * (1 + 0.34 * u2.to(ureg.meter/ureg.second).magnitude)
    
    et0 = (0.408 * numerator / denominator).to(ureg.millimeter / ureg.day)

    if output_path and dates is not None:
        df = pd.DataFrame({'date': dates, 'ET0(mm/day)': et0.magnitude})
        df.to_csv(output_path, index=False)
    
    return et0.magnitude
