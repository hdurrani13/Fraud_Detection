import json

import joblib

from fraud.data import make_synthetic
from fraud.train import run


def test_end_to_end_on_synthetic(tmp_path):
    report = run(make_synthetic(6000, fraud_rate=0.02, seed=3), tmp_path)

    assert report["best_model"] != "Baseline (always legit)"
    best = report["comparison"][report["best_model"]]["test"]
    baseline = report["comparison"]["Baseline (always legit)"]["test"]
    assert best["pr_auc"] > baseline["pr_auc"]
    # The chosen threshold should cost less than not using a model at all
    assert report["test_at_chosen_threshold"]["total_cost"] < report["test_with_no_model"]["total_cost"]

    saved = joblib.load(tmp_path / "models" / "model.joblib")
    assert {"model", "threshold", "features"} <= saved.keys()
    assert json.loads((tmp_path / "reports" / "metrics.json").read_text())["best_model"] == report["best_model"]
    for fig in ["pr_curves.png", "cost_curve.png", "feature_importance.png"]:
        assert (tmp_path / "reports" / "figures" / fig).stat().st_size > 1000
