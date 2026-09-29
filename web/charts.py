"""Draw every chart on the web page as a static SVG file (runs once, at build time).

Static images need no JavaScript, so the page shows its charts in any browser.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import CLUSTER_COLORS, REGIME_COLORS, REGIMES  # noqa: E402
from feature_engineering import FEATURE_LABELS  # noqa: E402

CHART_DIR = Path(__file__).resolve().parent / "static" / "charts"
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "svg.fonttype": "path",
})


def periods(df, column):
    block = (df[column] != df[column].shift()).cumsum()
    return df.groupby(block).agg(start=("date", "first"), end=("date", "last"), label=(column, "first"))


def shade(ax, bands, colors, alpha=0.3):
    step = pd.offsets.BDay(1)
    for row in bands.itertuples():
        ax.axvspan(row.start, row.end + step, color=colors[row.label], alpha=alpha, lw=0)


def save(fig, name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


def timeline(res, out_dir):
    data, test_start = res["data"], res["test"]["date"].min()
    names = res["cluster_profile"]["description"]
    cluster_label = {c: f"{c}: {d}" for c, d in names.items()}
    cluster_colors = {c: CLUSTER_COLORS[c] for c in names.index}

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1.6], "hspace": 0.45})
    shade(axes[0], periods(data, "true_regime"), REGIME_COLORS)
    axes[0].set_title("True regime (hidden from K-Means)")
    axes[0].legend(handles=[Patch(color=REGIME_COLORS[r], alpha=0.5, label=r) for r in REGIMES],
                   ncol=4, loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=9, handlelength=1.2)

    shade(axes[1], periods(data, "cluster"), cluster_colors, alpha=0.28)
    axes[1].set_title("K-Means clusters (no labels used)", pad=24)
    axes[1].legend(handles=[Patch(color=cluster_colors[c], alpha=0.5, label=cluster_label[c]) for c in names.index],
                   ncol=4, loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=8, handlelength=1.2,
                   columnspacing=1.0)

    for ax in axes[:2]:
        ax.plot(data["date"], data["close"], color=INK, lw=0.9)
        ax.set_ylabel("Price")
        ax.set_ylim(data["close"].min() * 0.95, data["close"].max() * 1.05)

    axes[2].fill_between(data["date"], data["rolling_volatility_20"] * 100, color=MUTED, alpha=0.45, lw=0)
    axes[2].plot(data["date"], data["rolling_volatility_20"] * 100, color="#374151", lw=0.7)
    axes[2].set_title("20-day volatility (%)")
    axes[2].set_ylim(bottom=0)

    for ax in axes:
        ax.axvline(test_start, color=MUTED, ls="--", lw=1)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_xlim(data["date"].min(), data["date"].max())
    axes[2].annotate("Test period →", (test_start, 1), xycoords=("data", "axes fraction"),
                     xytext=(5, -12), textcoords="offset points", color=MUTED, fontsize=9)
    save(fig, "timeline.svg", out_dir)


def heatmap(ax, table, cmap, xlabel, ylabel):
    values = table.values
    share = values / np.maximum(values.sum(axis=0 if cmap == "Blues" else 1, keepdims=True), 1)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, values[i, j], ha="center", va="center", fontsize=10,
                    color="white" if share[i, j] > 0.5 else INK)


def cluster_heatmap(res, out_dir):
    ct = res["crosstab"].reindex(columns=REGIMES, fill_value=0)
    names = res["cluster_profile"]["description"]
    ct.index = [f"{c}: {names[c]}" for c in ct.index]
    ct.index = [label.replace(", ", ",\n") for label in ct.index]
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    heatmap(ax, ct, "Blues", "True regime", None)
    ax.set_title("Where each true regime's days ended up")
    save(fig, "cluster_heatmap.svg", out_dir)


def confusion(res, out_dir):
    cm = res["test_metrics"]["confusion_matrix"].copy()
    cm.index = [i.replace("true ", "") for i in cm.index]
    cm.columns = [c.replace("pred ", "") for c in cm.columns]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    heatmap(ax, cm, "Greens", "Predicted", "True")
    ax.set_title("Confusion matrix (test period)")
    save(fig, "confusion.svg", out_dir)


def importance(res, out_dir):
    imp = res["importances"].rename(index=FEATURE_LABELS).sort_values()
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.barh(imp.index, imp.values * 100, color="#4f6d9a", height=0.6)
    for y, v in enumerate(imp.values * 100):
        ax.text(v + 0.6, y, f"{v:.0f}%", va="center", color=MUTED, fontsize=9)
    ax.set_title("What the model relies on (feature importance)")
    ax.set_xlabel("Importance (%)")
    ax.set_xlim(0, imp.max() * 118)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.tick_params(axis="y", length=0)
    save(fig, "importance.svg", out_dir)


def test_strip(res, out_dir):
    test = res["test"]
    fig, ax = plt.subplots(figsize=(11, 1.9))
    step = pd.offsets.BDay(1)
    for y, column in [(1, "true_regime"), (0, "predicted_regime")]:
        for row in periods(test, column).itertuples():
            ax.barh(y, (row.end + step) - row.start, left=row.start, height=0.75,
                    color=REGIME_COLORS[row.label], lw=0)
    ax.set_yticks([1, 0], ["True", "Predicted"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(test["date"].min(), test["date"].max() + step)
    ax.spines["left"].set_visible(False)
    ax.set_title("Test period: true regime vs prediction")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax.legend(handles=[Patch(color=REGIME_COLORS[r], label=r) for r in REGIMES],
              ncol=4, loc="lower right", bbox_to_anchor=(1, 1.02), fontsize=9)
    save(fig, "test_strip.svg", out_dir)


def build_all(res, out_dir=CHART_DIR):
    for draw in (timeline, cluster_heatmap, confusion, importance, test_strip):
        draw(res, out_dir)
    return sorted(p.name for p in out_dir.glob("*.svg"))


if __name__ == "__main__":
    from pipeline import load_or_run

    print("Charts written:", ", ".join(build_all(load_or_run())))
