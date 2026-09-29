"""Interactive dashboard: pick a threshold and see what it costs.

    streamlit run app/streamlit_app.py

Needs `python -m fraud.train` to have been run first (it writes models/ and reports/).
"""

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud.data import RAW_COLUMNS  # noqa: E402
from fraud.features import add_features  # noqa: E402
from fraud.metrics import cost_curve, evaluate_threshold  # noqa: E402

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")


@st.cache_resource
def load_artifacts():
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    report = json.loads((ROOT / "reports" / "metrics.json").read_text())
    scores = pd.read_csv(ROOT / "reports" / "test_scores.csv")
    return bundle, report, scores


if not (ROOT / "models" / "model.joblib").exists():
    st.error("No trained model found. Run `python -m fraud.train` first.")
    st.stop()

bundle, report, test = load_artifacts()
y, s, amt = test["Class"].to_numpy(), test["score"].to_numpy(), test["Amount"].to_numpy()

st.title("Credit-card fraud detection")
st.caption(
    f"{report['best_model']} scored {len(test):,} transactions from the last 20% of the dataset, "
    f"which the model never saw. {int(y.sum())} of them are fraud ({y.mean():.2%})."
)

with st.sidebar:
    st.header("Decision settings")
    review_cost = st.slider("Cost to review one flagged transaction ($)", 1, 50, int(report["review_cost_per_flag"]))
    default = float(round(report["threshold"], 3))
    picked = st.slider("Flag when fraud score is at least", 0.0, 1.0, default, step=0.001, format="%.3f")
    # The slider only has 3 decimals; use the exact saved threshold until the user moves it
    threshold = report["threshold"] if picked == default else picked
    st.caption(f"Cost-optimal threshold chosen on validation data: **{report['threshold']:.4f}**")

r = evaluate_threshold(y, s, amt, threshold, review_cost)
none = evaluate_threshold(y, s, amt, float("inf"), review_cost)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fraud caught", f"{r.true_positives} / {r.true_positives + r.false_negatives}", f"{r.recall:.0%} recall")
c2.metric("False alarms", f"{r.false_positives:,}", f"{r.precision:.0%} of flags are fraud", delta_color="off")
c3.metric("Fraud $ stopped", f"${r.fraud_dollars_caught:,.0f}", f"${r.fraud_dollars_missed:,.0f} missed", "inverse")
c4.metric(
    "Total cost",
    f"${r.total_cost:,.0f}",
    f"-${none.total_cost - r.total_cost:,.0f} vs no model",
    delta_color="inverse",
)

tab_cost, tab_flags, tab_models, tab_score = st.tabs(
    ["Cost trade-off", "Flagged transactions", "Model comparison", "Score your own file"]
)

with tab_cost:
    rows = [x.to_dict() for x in cost_curve(y, s, amt, review_cost)]
    curve = pd.DataFrame(rows).melt(
        id_vars=["threshold", "recall"],
        value_vars=["fraud_dollars_missed", "review_cost", "total_cost"],
        var_name="kind",
        value_name="dollars",
    )
    curve["kind"] = curve["kind"].map(
        {"fraud_dollars_missed": "Fraud missed", "review_cost": "Review cost", "total_cost": "Total"}
    )
    # Flagging everything costs far more than doing nothing; cap the axis so the useful range is visible
    y_max = none.total_cost * 1.4
    lines = (
        alt.Chart(curve)
        .mark_line(strokeWidth=2.5, clip=True)
        .encode(
            x=alt.X("recall:Q", title="Recall (share of fraud caught)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("dollars:Q", title="Dollars", scale=alt.Scale(domain=[0, y_max])),
            color=alt.Color(
                "kind:N",
                title=None,
                scale=alt.Scale(
                    domain=["Fraud missed", "Review cost", "Total"], range=["#d93025", "#2f6fdf", "#202124"]
                ),
            ),
            tooltip=[
                "kind",
                alt.Tooltip("recall:Q", format=".0%"),
                alt.Tooltip("threshold:Q", format=".3f"),
                alt.Tooltip("dollars:Q", format="$,.0f"),
            ],
        )
    )
    point = (
        alt.Chart(pd.DataFrame({"recall": [r.recall], "dollars": [r.total_cost]}))
        .mark_point(size=160, filled=True, color="#0f9d58")
        .encode(x="recall:Q", y="dollars:Q")
    )
    st.altair_chart((lines + point).properties(height=380), use_container_width=True)
    st.markdown(
        "Moving right catches more fraud (less **red**) but flags more transactions for review (more **blue**). "
        "The **green dot** is your current threshold; the cheapest point is the bottom of the black line."
    )

with tab_flags:
    flagged = test[s >= threshold].sort_values("score", ascending=False)
    flagged = flagged.assign(Result=flagged["Class"].map({1: "Fraud ✅", 0: "False alarm"}))
    st.write(f"{len(flagged):,} transactions flagged at this threshold.")
    st.dataframe(
        flagged[["score", "Amount", "Result"]].rename(columns={"score": "Fraud score"}).head(500),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fraud score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            "Amount": st.column_config.NumberColumn(format="$%.2f"),
        },
    )

with tab_models:
    rows = [
        {
            "Model": name,
            "PR-AUC": m["test"]["pr_auc"],
            "ROC-AUC": m["test"]["roc_auc"],
            "Recall at 90% precision": m["test"]["recall_at_90_precision"],
        }
        for name, m in report["comparison"].items()
    ]
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
    st.caption(
        "All models were trained on the first 64% of transactions (by time), compared on the next 16%, "
        "and scored here on the final 20%. PR-AUC is the main metric because fraud is so rare."
    )
    st.image(str(ROOT / "reports" / "figures" / "pr_curves.png"), width=650)

with tab_score:
    st.write(f"Upload a CSV with the columns `{RAW_COLUMNS[0]}`, `V1`–`V28` and `Amount`.")
    upload = st.file_uploader("Transactions CSV", type="csv")
    if upload:
        df = pd.read_csv(upload)
        missing = [c for c in RAW_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing[:6])}{'…' if len(missing) > 6 else ''}")
        else:
            X = add_features(df)[bundle["features"]]
            df.insert(0, "fraud_score", bundle["model"].predict_proba(X)[:, 1].round(4))
            df.insert(1, "flagged", df["fraud_score"] >= threshold)
            st.success(f"{int(df['flagged'].sum())} of {len(df):,} transactions flagged.")
            st.dataframe(df.sort_values("fraud_score", ascending=False), use_container_width=True, hide_index=True)
            st.download_button("Download scores", df.to_csv(index=False), "scored_transactions.csv", "text/csv")
