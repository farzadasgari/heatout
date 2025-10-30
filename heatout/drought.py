import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy import stats

def aggregate_to_monthly(df):
    monthly = df.resample('ME').agg({
        'total_precipitation_sum': 'sum',
        'tm': 'mean',
        'tmin': 'mean',
        'tmax': 'mean',
        'um': 'mean',
        'u_component_of_wind_10m': 'mean',
        'v_component_of_wind_10m': 'mean',
        'surface_net_solar_radiation_sum': 'sum',
        'surface_net_thermal_radiation_sum': 'sum',
        'surface_pressure': 'mean'
    })
    monthly['precip_mm'] = monthly['total_precipitation_sum'] * 1000
    monthly['temp_c'] = monthly['tm']
    monthly['tmin_c'] = monthly['tmin']
    monthly['tmax_c'] = monthly['tmax']
    monthly['rh_mean'] = monthly['um']
    monthly['wind_speed_10m'] = np.sqrt(monthly['u_component_of_wind_10m']**2 + monthly['v_component_of_wind_10m']**2)
    monthly['rn_mj_m2'] = (monthly['surface_net_solar_radiation_sum'] + monthly['surface_net_thermal_radiation_sum']) / 1e6
    monthly['pressure_kpa'] = monthly['surface_pressure'] / 1000
    
    return monthly

def thornthwaite_pet(monthly_temp, latitude, dates):
    num_months = len(monthly_temp)
    pet = np.zeros(num_months)
    
    mean_monthly_t = [np.mean(monthly_temp[month-1::12]) for month in range(1, 13) if len(monthly_temp[month-1::12]) > 0]
    i_monthly = np.maximum(0, np.array(mean_monthly_t) / 5)**1.514
    I = np.sum(i_monthly)
    
    a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239
    
    lat_rad = np.deg2rad(latitude)
    
    for idx in range(num_months):
        month_end = dates[idx] + pd.offsets.MonthEnd(0)
        ndm = (month_end - dates[idx].replace(day=1) + pd.Timedelta(1, 'D')).days
        
        j = dates[idx].dayofyear
        
        delta = 0.409 * np.sin(2 * np.pi / 365 * j - 1.39)
        ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))
        N = 24 / np.pi * ws
        
        K = (N / 12) * (ndm / 30.0)
        
        t = monthly_temp[idx]
        if t > 0:
            pet[idx] = 16 * K * (10 * t / I)**a
        else:
            pet[idx] = 0
    
    return pet

def penman_monteith_pet(daily_df, monthly_df):
    if 'ET0' in daily_df.columns:
        monthly_pet = daily_df['ET0'].resample('ME').sum()
    else:
        T = daily_df['tm']
        T_min = daily_df['tmin']
        T_max = daily_df['tmax']
        
        ws_10 = np.sqrt(daily_df['u_component_of_wind_10m']**2 + daily_df['v_component_of_wind_10m']**2)
        u2 = ws_10 * (4.87 / np.log(67.8 * 10 - 5.42))
        
        Rn = (daily_df['surface_net_solar_radiation_sum'] + daily_df['surface_net_thermal_radiation_sum']) / 1_000_000
        
        G = 0
        
        es_Tmax = 0.6108 * np.exp(17.27 * T_max / (T_max + 237.3))
        es_Tmin = 0.6108 * np.exp(17.27 * T_min / (T_min + 237.3))
        es = (es_Tmax + es_Tmin) / 2
        
        RH_mean = daily_df['um']
        ea = (RH_mean / 100) * es
        
        Delta = 4098 * (0.6108 * np.exp(17.27 * T / (T + 237.3))) / (T + 237.3)**2
        
        P = daily_df['surface_pressure'] / 1000
        gamma = 0.000665 * P
        
        numerator = 0.408 * Delta * (Rn - G) + gamma * (900 / (T + 273)) * u2 * (es - ea)
        denominator = Delta + gamma * (1 + 0.34 * u2)
        daily_df['ET0_mm_day'] = (numerator / denominator).clip(lower=0)
        
        monthly_pet = daily_df['ET0_mm_day'].resample('ME').sum()
    
    return monthly_pet.reindex(monthly_df.index).fillna(0).values

def compute_loglogistic_params(series):
    series = series[~np.isnan(series)]
    if len(series) < 3:
        return 1, 1, 0
    n = len(series)
    sorted_series = np.sort(series)
    ranks = np.arange(1, n+1)
    F = (ranks - 0.35) / n
    w0 = np.mean(sorted_series)
    w1 = np.sum((1 - F) * sorted_series) / n
    w2 = np.sum((1 - F)**2 * sorted_series) / n
    
    beta = (2 * w1 - w0) / (6 * w1 - w0 - 6 * w2)
    if beta == 0 or np.isnan(beta):
        beta = 1e-6
    try:
        gamma_ln = gammaln(1 + 1/beta) + gammaln(1 - 1/beta)
        alpha = (w0 - 2 * w1) * beta / np.exp(gamma_ln)
        gamma = w0 - alpha * np.exp(gamma_ln)
    except:
        alpha, beta, gamma = 1, 1, 0
    return alpha, beta, gamma

def loglogistic_cdf(x, alpha, beta, gamma):
    x = np.array(x)
    valid = x > gamma
    cdf = np.zeros_like(x)
    cdf[valid] = 1 / (1 + (alpha / (x[valid] - gamma))**beta)
    cdf[~valid] = 0
    return cdf

def spei_from_cdf(cdf):
    p = np.clip(cdf, 1e-10, 1 - 1e-10)
    spei = np.zeros_like(p)
    
    mask = p <= 0.5
    q = p[mask]
    w = np.sqrt(-2 * np.log(q))
    spei[mask] = -(w - (2.515517 + 0.802853 * w + 0.010328 * w**2) / (1 + 1.432788 * w + 0.189269 * w**2 + 0.001308 * w**3))
    
    mask = p > 0.5
    q = 1 - p[mask]
    w = np.sqrt(-2 * np.log(q))
    spei[mask] = w - (2.515517 + 0.802853 * w + 0.010328 * w**2) / (1 + 1.432788 * w + 0.189269 * w**2 + 0.001308 * w**3)
    
    return spei

def calculate_spei(monthly_df, daily_df, scales=[1, 3, 6, 12], latitude=30, pet_method='thornthwaite', calibration_period=None):
    if pet_method == 'thornthwaite':
        pet = thornthwaite_pet(monthly_df['temp_c'].values, latitude, monthly_df.index)
    elif pet_method == 'penman':
        pet = penman_monteith_pet(daily_df, monthly_df)
    else:
        raise ValueError("Invalid pet_method")
    
    d = monthly_df['precip_mm'].values - pet
    
    N = len(d)
    if calibration_period is None:
        cal_start_idx = 0
        cal_end_idx = N
    else:
        cal_start = pd.to_datetime(calibration_period[0])
        cal_end = pd.to_datetime(calibration_period[1])
        cal_start_idx = np.searchsorted(monthly_df.index, cal_start)
        cal_end_idx = np.searchsorted(monthly_df.index, cal_end)
    
    spei_dict = {}
    for k in scales:
        aggregated = np.convolve(d, np.ones(k), mode='valid')
        agg_index = monthly_df.index[k-1:]
        
        cal_agg_start = max(0, cal_start_idx - k + 1)
        cal_agg_end = min(len(aggregated), cal_end_idx - k + 1)
        cal_agg = aggregated[cal_agg_start:cal_agg_end]
        
        if len(cal_agg) < 3:
            cal_agg = aggregated
        
        alpha, beta, gamma = compute_loglogistic_params(cal_agg)
        
        cdf = loglogistic_cdf(aggregated, alpha, beta, gamma)
        spei = spei_from_cdf(cdf)
        
        spei_dict[k] = pd.Series(spei, index=agg_index)
    
    return spei_dict

def calculate_spi(monthly_df, scales=[1, 3, 6, 12], calibration_period=None):
    precip = monthly_df['precip_mm'].values
    N = len(precip)
    if calibration_period is None:
        cal_start_idx = 0
        cal_end_idx = N
    else:
        cal_start = pd.to_datetime(calibration_period[0])
        cal_end = pd.to_datetime(calibration_period[1])
        cal_start_idx = np.searchsorted(monthly_df.index, cal_start)
        cal_end_idx = np.searchsorted(monthly_df.index, cal_end)
    
    spi_dict = {}
    for k in scales:
        aggregated = np.convolve(precip, np.ones(k), mode='valid')
        agg_index = monthly_df.index[k-1:]
        
        cal_agg_start = max(0, cal_start_idx - k + 1)
        cal_agg_end = min(len(aggregated), cal_end_idx - k + 1)
        cal_agg = aggregated[cal_agg_start:cal_agg_end]
        
        if len(cal_agg) < 3:
            cal_agg = aggregated
        
        pos = cal_agg > 0
        p_zero = np.mean(cal_agg == 0)
        if np.sum(pos) > 0:
            shape, loc, scale = stats.gamma.fit(cal_agg[pos])
        else:
            shape, loc, scale = 1, 0, 1
        
        cdf = np.full(len(aggregated), p_zero)
        pos_now = aggregated > 0
        cdf[pos_now] = p_zero + (1 - p_zero) * stats.gamma.cdf(aggregated[pos_now], shape, loc=loc, scale=scale)
        
        spi = stats.norm.ppf(np.clip(cdf, 0.001, 0.999))
        
        spi_dict[k] = pd.Series(spi, index=agg_index)
    
    return spi_dict

def palmer_drought_severity_index(monthly_precip, monthly_temp, awc=100, latitude=30, dates=None):
    if dates is None:
        dates = pd.date_range(start='2014-04-30', periods=len(monthly_precip), freq='ME')
    pet = thornthwaite_pet(monthly_temp, latitude, dates)
    num_months = len(monthly_precip)
    soil_moisture = np.zeros(num_months)
    soil_moisture[0] = awc / 2
    runoff = np.zeros(num_months)
    recharge = np.zeros(num_months)
    for i in range(1, num_months):
        avail = soil_moisture[i-1] + monthly_precip[i]
        if avail > pet[i]:
            runoff[i] = avail - pet[i]
            soil_moisture[i] = min(awc, soil_moisture[i-1] + monthly_precip[i] - pet[i])
        else:
            recharge[i] = pet[i] - avail
            soil_moisture[i] = 0
    d = monthly_precip - pet
    k = 1.0
    z_index = k * d
    pdsi = np.zeros(num_months)
    pdsi[0] = z_index[0] / 3
    for i in range(1, num_months):
        pdsi[i] = 0.897 * pdsi[i-1] + z_index[i] / 3
    return pdsi

def add_drought_indices_to_daily(df, latitude=30, scales=[1,3,6,12], calibration_start='2014-01-01', calibration_end='2025-01-01', awc=100):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    monthly = aggregate_to_monthly(df)
    cal_period = (calibration_start, calibration_end)
    
    spei_thorn = calculate_spei(monthly, df, scales, latitude, pet_method='thornthwaite', calibration_period=cal_period)
    spei_penman = calculate_spei(monthly, df, scales, latitude, pet_method='penman', calibration_period=cal_period)
    spi = calculate_spi(monthly, scales, calibration_period=cal_period)
    
    monthly_precip = monthly['precip_mm'].values
    monthly_temp = monthly['temp_c'].values
    pdsi_monthly = palmer_drought_severity_index(monthly_precip, monthly_temp, awc, latitude, monthly.index)
    pdsi_series = pd.Series(pdsi_monthly, index=monthly.index).resample('D').ffill().reindex(df.index, method='nearest')
    df['PDSI'] = pdsi_series
    
    for k in scales:
        monthly_spei_thorn = spei_thorn[k].resample('D').ffill().reindex(df.index, method='nearest')
        df[f'SPEI_thorn_{k}'] = monthly_spei_thorn
        
        monthly_spei_penman = spei_penman[k].resample('D').ffill().reindex(df.index, method='nearest')
        df[f'SPEI_penman_{k}'] = monthly_spei_penman
        
        monthly_spi = spi[k].resample('D').ffill().reindex(df.index, method='nearest')
        df[f'SPI_{k}'] = monthly_spi
    
    return df

# Load the dataset
df = pd.read_csv('../dataset/daily.csv')

# Calculate drought indices
df_with_indices = add_drought_indices_to_daily(
    df,
    latitude=30.8,
    scales=[1, 3, 6, 12],
    calibration_start='2014-01-01',
    calibration_end='2025-01-01',
    awc=150
)
print(df_with_indices.head())
df_with_indices.to_csv('../dataset/drought.csv')
