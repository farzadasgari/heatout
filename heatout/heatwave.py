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
