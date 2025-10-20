import numpy as np
from scipy.stats import norm


def constant_threshold(daily_temps, threshold=35.0, duration=3):
    exceed = daily_temps > threshold
    heatwave = np.zeros_like(daily_temps, dtype=int)
    count = 0
    for i in range(len(daily_temps)):
        if exceed[i]:
            count += 1
            if count >= duration:
                heatwave[i - count + 1:i + 1] = 1
        else:
            count = 0
    return heatwave


def average_plus5(daily_temps, duration=3):
    mean_temp = np.mean(daily_temps)
    threshold = mean_temp + 5.0
    return constant_threshold(daily_temps, threshold, duration)


def upper_tail_percentile(daily_temps, percentile=90, window=15, duration=3):
    num_days = len(daily_temps)
    days_per_year = 365
    num_years = num_days // days_per_year
    temps_2d = daily_temps.reshape(num_years, days_per_year)
    
    thresholds = np.zeros(days_per_year)
    for doy in range(days_per_year):
        start = max(0, doy - window // 2)
        end = min(days_per_year, doy + window // 2 + 1)
        local_temps = temps_2d[:, start:end].flatten()
        thresholds[doy] = np.percentile(local_temps, percentile)
    
    full_thresholds = np.tile(thresholds, num_years)
    return constant_threshold(daily_temps, full_thresholds, duration)


def summer_derived_threshold(daily_temps, percentile=90, summer_start_doy=152, summer_end_doy=243, duration=3):
    num_days = len(daily_temps)
    days_per_year = 365
    num_years = num_days // days_per_year
    temps_2d = daily_temps.reshape(num_years, days_per_year)
    
    summer_temps = temps_2d[:, summer_start_doy-1:summer_end_doy].flatten()
    threshold = np.percentile(summer_temps, percentile)
    return constant_threshold(daily_temps, threshold, duration)
