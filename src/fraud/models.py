"""Candidate models. All handle class imbalance with class weights rather than resampling."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def candidate_models(seed: int = 42) -> dict:
    return {
        "Baseline (always legit)": DummyClassifier(strategy="prior"),
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight={0: 1, 1: 10},
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        ),
    }
