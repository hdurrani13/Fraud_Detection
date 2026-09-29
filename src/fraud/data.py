"""Loading, validating and splitting the credit-card transactions dataset.

Dataset: "Credit Card Fraud Detection" (ULB Machine Learning Group, Kaggle).
284,807 European card transactions over two days in September 2013, 492 of
them fraudulent. V1-V28 are PCA components (anonymised by the bank); only
Time (seconds since the first transaction) and Amount are raw.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "data" / "creditcard.csv"

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_COLUMNS = ["Time", *PCA_COLUMNS, "Amount"]
TARGET = "Class"


def load_transactions(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and put it in data/."
        )
    df = pd.read_csv(path)
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in [*RAW_COLUMNS, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError("Class must be 0 (legit) or 1 (fraud)")


def time_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: train on earlier transactions, test on later ones.

    A random split would let the model see "the future", which overstates how
    well it would work in production, where it only ever scores new transactions.
    """
    ordered = df.sort_values("Time", kind="stable").reset_index(drop=True)
    cut = int(len(ordered) * (1 - test_fraction))
    return ordered.iloc[:cut].reset_index(drop=True), ordered.iloc[cut:].reset_index(drop=True)


def make_synthetic(n: int = 5000, fraud_rate: float = 0.01, seed: int = 0) -> pd.DataFrame:
    """Random data with the real schema, for tests and CI (the real file isn't in the repo)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    X = rng.normal(size=(n, 28))
    # Make a few components informative so models have something to learn
    X[y == 1, :4] += rng.normal(2.5, 1.0, size=(y.sum(), 4))
    amount = np.round(rng.lognormal(3.5, 1.2, n), 2)
    amount[y == 1] = np.round(rng.lognormal(4.2, 1.3, y.sum()), 2)
    df = pd.DataFrame(X, columns=PCA_COLUMNS)
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_800, n)).round())
    df["Amount"] = amount
    df[TARGET] = y
    return df
