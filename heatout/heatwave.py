import numpy as np
from scipy.stats import norm


def constant_threshold(daily_temps, threshold=35.0, duration=3):
    """
    Detects heatwaves where temperatures exceed a fixed threshold for at least 'duration' consecutive days.
    - daily_temps: np.array of daily temperatures (e.g., TMAX, TMIN, or TMEAN).
    - threshold: Fixed temperature threshold in °C.
    - duration: Minimum consecutive days (temporal persistence).
    Returns: Binary np.array (1 if heatwave day, 0 otherwise).
    """
    exceed = daily_temps > threshold
    heatwave = np.zeros_like(daily_temps, dtype=int)
    count = 0
    for i in range(len(daily_temps)):
        if exceed[i]:
            count += 1
            if count >= duration:
                heatwave[i - count + 1:i + 1] = 1  # Mark the entire run
        else:
            count = 0
    return heatwave
