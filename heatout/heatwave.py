import pandas as pd
import numpy as np
from scipy.stats import norm

# Load the drought.csv file
df = pd.read_csv('../dataset/drought.csv')

# Ensure Date column is in datetime format and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Define heatwave index functions with leap year handling
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

def upper_tail_percentile(daily_temps, dates, percentile=90, window=15, duration=3):
    num_days = len(daily_temps)
    thresholds = np.zeros(num_days)
    
    for i in range(num_days):
        doy = dates[i].dayofyear
        start_idx = max(0, i - window // 2)
        end_idx = min(num_days, i + window // 2 + 1)
        local_temps = daily_temps[start_idx:end_idx]
        thresholds[i] = np.percentile(local_temps, percentile)
    
    return constant_threshold(daily_temps, thresholds, duration)

def summer_derived_threshold(daily_temps, dates, percentile=90, summer_start_doy=152, summer_end_doy=243, duration=3):
    summer_mask = (dates.dayofyear >= summer_start_doy) & (dates.dayofyear <= summer_end_doy)
    summer_temps = daily_temps[summer_mask]
    threshold = np.percentile(summer_temps, percentile)
    return constant_threshold(daily_temps, threshold, duration)

def excess_heat_factor(daily_temps, ehf_threshold=0.0):
    t95 = np.percentile(daily_temps, 95)
    ehf_values = np.zeros(len(daily_temps))
    for i in range(2, len(daily_temps)):
        three_day_avg = np.mean(daily_temps[i-2:i+1])
        ehi_sig = three_day_avg - t95
        prior_avg = np.mean(daily_temps[max(0, i-32):i-2]) if i > 2 else 0
        ehi_accl = three_day_avg - prior_avg
        ehf_values[i] = ehi_sig * max(1, ehi_accl)
    binary = (ehf_values >= ehf_threshold).astype(int)
    return ehf_values, binary

def standardized_heat_index(daily_temps, dates, window=15, shi_threshold=1.0):
    num_days = len(daily_temps)
    shi_values = np.zeros(num_days)
    
    for i in range(2, num_days):
        doy = dates[i].dayofyear
        start_idx = max(0, i - window // 2)
        end_idx = min(num_days, i + window // 2 + 1)
        local_temps = daily_temps[start_idx:end_idx]
        three_day_avg = np.mean(daily_temps[i-2:i+1])
        
        sorted_temps = np.sort(local_temps)
        ranks = np.searchsorted(sorted_temps, three_day_avg) + 1
        n = len(local_temps)
        p = (ranks - 0.44) / (n + 0.12)
        shi_values[i] = norm.ppf(p) if 0 < p < 1 else 0
    
    binary = (shi_values >= shi_threshold).astype(int)
    return shi_values, binary

# Calculate heatwave indices using 'tm' (mean temperature) as daily_temps
daily_temps = df['tm'].values
dates = df.index

df['Heatwave_Constant'] = constant_threshold(daily_temps, threshold=35.0, duration=3)
df['Heatwave_AvgPlus5'] = average_plus5(daily_temps, duration=3)
df['Heatwave_UpperPercentile'] = upper_tail_percentile(daily_temps, dates, percentile=90, window=15, duration=3)
df['Heatwave_SummerDerived'] = summer_derived_threshold(daily_temps, dates, percentile=90, summer_start_doy=152, summer_end_doy=243, duration=3)
ehf_values, df['Heatwave_EHF_Binary'] = excess_heat_factor(daily_temps, ehf_threshold=0.0)
df['Heatwave_EHF_Value'] = ehf_values
shi_values, df['Heatwave_SHI_Binary'] = standardized_heat_index(daily_temps, dates, window=15, shi_threshold=1.0)
df['Heatwave_SHI_Value'] = shi_values

# Reset index to include Date column for export
df.reset_index(inplace=True)

# Save to heat.csv
df.to_csv('../dataset/heat.csv', index=False)
print("Heatwave indices added and saved to 'heat.csv'")
