"""Generate a reproducible synthetic daily market dataset with four hidden regimes."""
import numpy as np
import pandas as pd

from config import DATA_PATH, RANDOM_STATE, REGIMES

# Per-regime settings: (mean daily return, daily return std-dev, volume multiplier)
REGIME_PARAMS = {
    "Bull": (0.0007, 0.008, 1.05),
    "Bear": (-0.0006, 0.012, 1.20),
    "Stable": (0.0001, 0.005, 0.90),
    "Volatile": (0.0000, 0.025, 1.55),
}
REGIME_PROBS = [0.30, 0.20, 0.30, 0.20]


def generate(n_days=2500, seed=RANDOM_STATE, n_missing=25):
    """Return a DataFrame of synthetic prices, returns, volume and the true regime.

    Regimes arrive in blocks of 60-220 trading days. A few volume values are
    blanked out on purpose so the cleaning step has something to do.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n_days)

    regime_blocks = []
    while sum(map(len, regime_blocks)) < n_days:
        regime = rng.choice(REGIMES, p=REGIME_PROBS)
        length = int(rng.integers(60, 220))
        regime_blocks.append([regime] * length)
    true_regime = np.array(sum(regime_blocks, []))[:n_days]

    returns = np.empty(n_days)
    volume = np.empty(n_days)
    for i, regime in enumerate(true_regime):
        mu, sigma, volume_mult = REGIME_PARAMS[regime]
        returns[i] = rng.normal(mu, sigma)
        volume[i] = rng.lognormal(mean=13.0, sigma=0.22) * volume_mult

    df = pd.DataFrame({
        "date": dates,
        "close": 100 * np.exp(np.cumsum(returns)),
        "daily_return": returns,
        "volume": volume,
        "true_regime": true_regime,
    })

    missing_idx = rng.choice(n_days, size=n_missing, replace=False)
    df.loc[missing_idx, "volume"] = np.nan
    return df


def save(df, path=DATA_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    data = generate()
    out = save(data)
    print(f"Saved {len(data)} rows to {out.relative_to(out.parents[1])}")
