"""Turn raw prices and volume into regime-level features.

Every feature only looks backwards in time (rolling or expanding windows), so a
value on day t uses nothing from day t+1 onwards.
"""
import numpy as np

# Used by the Random Forest classifier.
FEATURES = [
    "rolling_mean_20",        # average daily return over the last 20 days (short-term trend)
    "rolling_volatility_20",  # std-dev of daily returns over the last 20 days (short-term risk)
    "rolling_mean_60",        # average daily return over the last 60 days (longer trend)
    "rolling_volatility_60",  # std-dev of daily returns over the last 60 days
    "momentum_20",            # % price change over the last 20 days
    "relative_volume_20",     # 20-day average volume / long-run median volume so far
]

# Used by K-Means. A smaller set of slow-moving features, so clusters describe
# market conditions rather than one-day noise.
CLUSTER_FEATURES = [
    "rolling_mean_60",
    "rolling_volatility_20",
    "momentum_20",
    "relative_volume_20",
]

FEATURE_LABELS = {
    "rolling_mean_20": "20-day mean return",
    "rolling_volatility_20": "20-day volatility",
    "rolling_mean_60": "60-day mean return",
    "rolling_volatility_60": "60-day volatility",
    "momentum_20": "20-day momentum",
    "relative_volume_20": "Relative volume (20-day)",
}


def add_features(df):
    out = df.copy()
    ret, vol, close = out["daily_return"], out["volume"], out["close"]

    out["rolling_mean_20"] = ret.rolling(20).mean()
    out["rolling_volatility_20"] = ret.rolling(20).std()
    out["rolling_mean_60"] = ret.rolling(60).mean()
    out["rolling_volatility_60"] = ret.rolling(60).std()
    out["momentum_20"] = close.pct_change(20)
    # Volume level matters (turbulent markets trade more), but raw volume drifts
    # over time in real data, so compare it with its own history instead.
    out["relative_volume_20"] = vol.rolling(20).mean() / vol.expanding(min_periods=20).median()

    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=FEATURES).reset_index(drop=True)
