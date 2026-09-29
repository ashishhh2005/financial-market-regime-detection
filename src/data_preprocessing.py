"""Load and clean the market dataset."""
import pandas as pd

from config import DATA_PATH


def clean(df):
    """Sort by date, drop duplicate dates and fill missing volume.

    Missing volume is forward-filled (carry the last known value), so no value
    is ever filled using information from a later day. Only a gap on the very
    first row falls back to the next available value.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    out["volume"] = out["volume"].ffill().bfill()
    return out


def load_and_clean(path=DATA_PATH):
    return clean(pd.read_csv(path))
