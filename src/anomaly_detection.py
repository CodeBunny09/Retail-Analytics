"""Static and temporal anomaly detection."""
import pandas as pd
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Static outliers per Dept/Store using Z‑score
# ---------------------------------------------------------------------------

def detect_static_outliers(df: pd.DataFrame, z_thresh: float = 3.5):
    z = np.abs(stats.zscore(df["Weekly_Sales"]))
    return df.assign(Outlier=z > z_thresh)

# ---------------------------------------------------------------------------
# Temporal anomalies using rolling window Z‑score
# ---------------------------------------------------------------------------

def detect_time_anomalies(df: pd.DataFrame, window: int = 12, z_thresh: float = 3):
    sales = df.set_index("Date")["Weekly_Sales"].rolling(window).mean()
    z = np.abs((df["Weekly_Sales"] - sales) / df["Weekly_Sales"].rolling(window).std())
    return df.assign(TimeOutlier=z > z_thresh)