import numpy as np
import pytest

from fraud.data import make_synthetic, time_split, validate
from fraud.features import FEATURES, add_features


def test_synthetic_matches_schema():
    df = make_synthetic(500)
    validate(df)
    assert df["Class"].isin([0, 1]).all()


def test_validate_rejects_missing_columns():
    with pytest.raises(ValueError):
        validate(make_synthetic(50).drop(columns=["V3"]))


def test_time_split_is_chronological_and_complete():
    df = make_synthetic(1000).sample(frac=1, random_state=1)  # shuffle
    train, test = time_split(df, 0.25)
    assert len(train) + len(test) == len(df)
    assert len(test) == 250
    assert train["Time"].max() <= test["Time"].min()


def test_features_added():
    df = add_features(make_synthetic(100))
    assert set(FEATURES) <= set(df.columns)
    assert np.allclose(df["hour_sin"] ** 2 + df["hour_cos"] ** 2, 1)
    assert (df["log_amount"] >= 0).all()


def test_midnight_and_late_evening_are_close():
    import pandas as pd

    df = pd.DataFrame({"Time": [23.9 * 3600, 0.1 * 3600, 12 * 3600], "Amount": [1, 1, 1]})
    f = add_features(df)
    near = np.hypot(f.hour_sin[0] - f.hour_sin[1], f.hour_cos[0] - f.hour_cos[1])
    far = np.hypot(f.hour_sin[0] - f.hour_sin[2], f.hour_cos[0] - f.hour_cos[2])
    assert near < far
