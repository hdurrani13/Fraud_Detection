"""Train and compare models, pick a cost-optimal threshold, and save everything.

    python -m fraud.train               # real data in data/creditcard.csv
    python -m fraud.train --synthetic   # fake data, for a quick smoke test

Splits (all chronological):
    fit (64%)  -> train candidate models
    val (16%)  -> choose the model and the decision threshold
    test (20%) -> final numbers, touched once
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import plots
from .data import ROOT, TARGET, load_transactions, make_synthetic, time_split
from .features import FEATURES, add_features
from .metrics import DEFAULT_REVIEW_COST, best_threshold, cost_curve, evaluate_threshold, summary_metrics
from .models import candidate_models


def score(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def run(df: pd.DataFrame, out_dir: Path, review_cost: float = DEFAULT_REVIEW_COST, seed: int = 42) -> dict:
    df = add_features(df)
    train, test = time_split(df, 0.2)
    fit, val = time_split(train, 0.2)
    X_fit, y_fit = fit[FEATURES], fit[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    comparison, test_scores, fitted = {}, {}, {}
    for name, model in candidate_models(seed).items():
        model.fit(X_fit, y_fit)
        fitted[name] = model
        val_scores = score(model, X_val)
        test_scores[name] = score(model, X_test)
        comparison[name] = {
            "val": summary_metrics(y_val, val_scores),
            "test": summary_metrics(y_test, test_scores[name]),
        }
        print(f"{name:26s} val PR-AUC {comparison[name]['val']['pr_auc']:.3f}")

    # Model selection uses validation data only
    best_name = max(comparison, key=lambda n: comparison[n]["val"]["pr_auc"])
    best = fitted[best_name]
    val_scores = score(best, X_val)
    chosen = best_threshold(y_val, val_scores, val["Amount"], review_cost)

    final = evaluate_threshold(y_test, test_scores[best_name], test["Amount"], chosen.threshold, review_cost)
    default = evaluate_threshold(y_test, test_scores[best_name], test["Amount"], 0.5, review_cost)
    no_model = evaluate_threshold(y_test, np.zeros(len(test)), test["Amount"], 1.0, review_cost)

    # Permutation importance on a sample of validation data
    sample = val.sample(min(len(val), 20_000), random_state=seed)
    imp = permutation_importance(
        best, sample[FEATURES], sample[TARGET], scoring="average_precision", n_repeats=3, random_state=seed, n_jobs=-1
    )

    report = {
        "rows": {"fit": len(fit), "val": len(val), "test": len(test)},
        "fraud_rate": {"fit": float(y_fit.mean()), "val": float(y_val.mean()), "test": float(y_test.mean())},
        "review_cost_per_flag": review_cost,
        "comparison": comparison,
        "best_model": best_name,
        "threshold": chosen.threshold,
        "test_at_chosen_threshold": final.to_dict(),
        "test_at_default_0.5": default.to_dict(),
        "test_with_no_model": no_model.to_dict(),
        "top_features": sorted(zip(FEATURES, imp.importances_mean.round(4).tolist(), strict=True), key=lambda x: -x[1])[
            :10
        ],
    }

    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": best, "name": best_name, "threshold": chosen.threshold, "features": FEATURES},
        out_dir / "models" / "model.joblib",
    )
    (out_dir / "reports" / "metrics.json").write_text(json.dumps(report, indent=2))
    test.assign(score=test_scores[best_name])[["Time", "Amount", TARGET, "score"]].to_csv(
        out_dir / "reports" / "test_scores.csv", index=False
    )

    figs = out_dir / "reports" / "figures"
    plots.pr_curves(test_scores, y_test, figs / "pr_curves.png")
    plots.cost_curve_plot(
        cost_curve(y_test, test_scores[best_name], test["Amount"], review_cost), final, default, figs / "cost_curve.png"
    )
    plots.importance_plot(FEATURES, imp.importances_mean, figs / "feature_importance.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=None, help="Path to creditcard.csv")
    parser.add_argument("--synthetic", action="store_true", help="Use generated data instead of the real file")
    parser.add_argument("--review-cost", type=float, default=DEFAULT_REVIEW_COST, help="Dollars per flagged review")
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()

    df = make_synthetic(20_000) if args.synthetic else load_transactions(args.data or ROOT / "data" / "creditcard.csv")
    report = run(df, args.out, args.review_cost)

    r = report["test_at_chosen_threshold"]
    none = report["test_with_no_model"]
    print(f"\nBest model: {report['best_model']} (threshold {report['threshold']:.3f})")
    print(f"Test PR-AUC: {report['comparison'][report['best_model']]['test']['pr_auc']:.3f}")
    print(
        f"Caught {r['true_positives']}/{r['true_positives'] + r['false_negatives']} frauds "
        f"with {r['false_positives']} false alarms (precision {r['precision']:.1%}, recall {r['recall']:.1%})"
    )
    print(f"Cost: ${r['total_cost']:,.0f} vs ${none['total_cost']:,.0f} with no model")


if __name__ == "__main__":
    main()
