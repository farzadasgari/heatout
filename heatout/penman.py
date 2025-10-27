import numpy as np
import pandas as pd

data = pd.read_csv("../dataset/merged_field_era.csv")

def calculate_evapotranspiration(df):
    T = df['tm']
    T_min = df['tmin']
    T_max = df['tmax']
    es_Tmax = 0.6108 * np.exp(17.27 * T_max / (T_max + 237.3))
    es_Tmin = 0.6108 * np.exp(17.27 * T_min / (T_min + 237.3))
    es = (es_Tmax + es_Tmin) / 2
    RH_mean = df['um']
    ea = (RH_mean / 100) * es
    ws_10 = np.sqrt(df['u_component_of_wind_10m']**2 + df['v_component_of_wind_10m']**2)
    u2 = ws_10 * (4.87 / np.log(67.8 * 10 - 5.42))  # ≈ ws_10 * 0.748
    Rn = (df['surface_net_solar_radiation_sum'] + df['surface_net_thermal_radiation_sum']) / 1_000_000
    G = 0
    Delta = 4098 * (0.6108 * np.exp(17.27 * T / (T + 237.3))) / (T + 237.3)**2
    P = df['surface_pressure'] / 1000  # Pa to kPa
    gamma = 0.000665 * P
    numerator = 0.408 * Delta * (Rn - G) + gamma * (900 / (T + 273)) * u2 * (es - ea)
    denominator = Delta + gamma * (1 + 0.34 * u2)
    et0 = numerator / denominator
    et0 = et0.clip(lower=0)
    return et0

data['ET0'] = calculate_evapotranspiration(data)
data.to_csv("../dataset/daily.csv", index=False)
