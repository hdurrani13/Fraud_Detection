import numpy as np
import pandas as pd

from .data import PCA_COLUMNS

ENGINEERED = ["log_amount", "hour_sin", "hour_cos"]
FEATURES = [*PCA_COLUMNS, *ENGINEERED]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features derived from the two raw columns.

    - log_amount: amounts are heavily right-skewed; the log is easier for linear models.
    - hour_sin/cos: time of day, encoded on a circle so 23:00 and 00:00 are close.
      Time is seconds since the first transaction, so this assumes the data starts
      around midnight, which is approximately true for this dataset.
    """
    out = df.copy()
    out["log_amount"] = np.log1p(out["Amount"])
    hour = (out["Time"] % 86_400) / 3_600
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out
