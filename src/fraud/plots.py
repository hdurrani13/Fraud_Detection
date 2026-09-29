import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

GREEN, BLUE, GREY, RED = "#0f9d58", "#2f6fdf", "#8a8f98", "#d93025"
PALETTE = [GREY, BLUE, "#e8710a", GREEN]


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def pr_curves(results: dict[str, np.ndarray], y, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for (name, scores), color in zip(results.items(), PALETTE, strict=False):
        if np.unique(scores).size == 1:
            continue  # a constant-score baseline has no curve; shown as the dashed line instead
        p, r, _ = precision_recall_curve(y, scores)
        ax.plot(r, p, label=name, color=color, lw=2, drawstyle="steps-post")
    ax.axhline(np.mean(y), color=GREY, ls="--", lw=1.5, label=f"Random guessing ({np.mean(y):.2%})")
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of flags that are fraud)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower left")
    _style(ax, "Precision-recall on the test period")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cost_curve_plot(curve, chosen, default, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    recalls = [r.recall for r in curve]
    ax.plot(recalls, [r.fraud_dollars_missed for r in curve], color=RED, lw=2, label="Fraud missed ($)")
    ax.plot(recalls, [r.review_cost for r in curve], color=BLUE, lw=2, label="Review cost ($)")
    ax.plot(recalls, [r.total_cost for r in curve], color="black", lw=2.5, label="Total cost ($)")
    ax.scatter([chosen.recall], [chosen.total_cost], color=GREEN, s=70, zorder=5, label="Chosen threshold")
    ax.scatter([default.recall], [default.total_cost], color=GREY, s=50, zorder=5, marker="s", label="Default 0.5")
    no_model = curve[-1].total_cost  # threshold = inf flags nothing
    ax.set_ylim(0, no_model * 1.4)  # flagging everything costs far more; keep the useful range visible
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Dollars (test period)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    ax.legend(frameon=False)
    _style(ax, "Cost of each decision threshold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def importance_plot(names, values, path, top: int = 12):
    order = np.argsort(values)[-top:]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(np.array(names)[order], np.array(values)[order], color=GREEN)
    ax.set_xlabel("Drop in PR-AUC when shuffled")
    _style(ax, "Most important features (permutation)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
