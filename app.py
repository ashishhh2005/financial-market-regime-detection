"""Market Regime Detection - interactive dashboard (Streamlit).

Run locally:   streamlit run app.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from config import CLUSTER_COLORS, RANDOM_STATE, REGIME_COLORS, REGIMES  # noqa: E402
from feature_engineering import FEATURE_LABELS, FEATURES  # noqa: E402
from pipeline import load_or_run  # noqa: E402

st.set_page_config(page_title="Market Regime Detection", page_icon="📈", layout="wide")

REPO_URL = os.environ.get("REPO_URL", "").strip()
REGIME_SCALE = alt.Scale(domain=REGIMES, range=[REGIME_COLORS[r] for r in REGIMES])
# How each feature is shown to people: multiply by `scale`, then add the unit.
DISPLAY = {
    "rolling_mean_20": (100, "%", "%.3f"),
    "rolling_volatility_20": (100, "%", "%.2f"),
    "rolling_mean_60": (100, "%", "%.3f"),
    "rolling_volatility_60": (100, "%", "%.2f"),
    "momentum_20": (100, "%", "%.1f"),
    "relative_volume_20": (1, "×", "%.2f"),
}
FEATURE_HELP = {
    "rolling_mean_20": "Average daily return over the last 20 trading days.",
    "rolling_volatility_20": "How much daily returns swung over the last 20 trading days (standard deviation).",
    "rolling_mean_60": "Average daily return over the last 60 trading days.",
    "rolling_volatility_60": "Standard deviation of daily returns over the last 60 trading days.",
    "momentum_20": "Price change over the last 20 trading days.",
    "relative_volume_20": "Last 20 days' average volume compared with the typical volume so far (1.0 = normal).",
}


@st.cache_resource(show_spinner="Generating market data and training the models…", max_entries=3)
def get_results(seed: int):
    return load_or_run(seed)


def periods(df, column):
    """Collapse consecutive days with the same label into (start, end, label) bands."""
    block = (df[column] != df[column].shift()).cumsum()
    return (
        df.groupby(block)
        .agg(start=("date", "first"), end=("date", "last"), label=(column, "first"), days=("date", "size"))
        .reset_index(drop=True)
    )


def pct(x, digits=1):
    return f"{x * 100:.{digits}f}%"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    seed = int(st.number_input(
        "Dataset seed", min_value=0, max_value=9999, value=RANDOM_STATE, step=1,
        help="Each seed creates a different synthetic market history. Everything is regenerated and retrained.",
    ))
    if seed != RANDOM_STATE:
        st.caption("New seeds are trained live, which can take up to a minute on the free server.")
    st.divider()
    st.markdown(
        "**About**  \nAn end-to-end data science project: synthetic market data, feature "
        "engineering, unsupervised clustering (K-Means) and supervised prediction (Random Forest)."
    )
    if REPO_URL:
        st.markdown(f"[Source code on GitHub]({REPO_URL})")
    st.caption("Educational project on synthetic data. Not investment advice.")

res = get_results(seed)
data, test = res["data"], res["test"]
metrics, cv = res["test_metrics"], res["cv"]
profile = res["cluster_profile"]
cluster_names = {c: f"{c}: {d}" for c, d in profile["description"].items()}
cluster_domain = [cluster_names[c] for c in sorted(cluster_names)]
CLUSTER_SCALE = alt.Scale(domain=cluster_domain, range=CLUSTER_COLORS[: len(cluster_domain)])
data = data.assign(cluster_name=data["cluster"].map(cluster_names))
test_start = test["date"].min()

# ---------------------------------------------------------------- header
st.title("Market Regime Detection")
st.markdown(
    "Can a model tell whether a market is in a **Bull**, **Bear**, **Stable** or **Volatile** phase "
    "from its recent behaviour? This dashboard generates a synthetic market with hidden regimes, "
    "tries to *discover* them without labels (K-Means), then *predicts* them with a Random Forest "
    "tested on dates it has never seen."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Test accuracy", pct(metrics["accuracy"]),
    f"+{(metrics['accuracy'] - res['baseline_accuracy']) * 100:.0f} pts vs. baseline",
    help=f"Newest 20% of dates ({len(test)} days), never seen in training. "
         f"Baseline = always guess the most common regime ({pct(res['baseline_accuracy'])}).",
)
k2.metric(
    "Cross-validated accuracy", pct(cv["accuracy"].mean()),
    f"range {pct(cv['accuracy'].min(), 0)}–{pct(cv['accuracy'].max(), 0)}", delta_color="off", delta_arrow="off",
    help="Five consecutive time blocks, each tested once, with a 60-day gap around it so rolling windows can't leak.",
)
k3.metric(
    "Cluster agreement (ARI)", f"{res['ari']:.2f}",
    help="Adjusted Rand index between K-Means clusters and the hidden true regimes. 1 = perfect match, 0 = random.",
)
k4.metric(
    "Trading days", f"{len(data):,}",
    f"{len(periods(data, 'true_regime'))} regime periods", delta_color="off", delta_arrow="off",
    help=f"{res['raw_rows']:,} generated days; the first rows are used up by 60-day rolling windows. "
         f"{res['missing_volume_filled']} missing volume values were filled.",
)

tab_timeline, tab_cluster, tab_model, tab_try, tab_how = st.tabs(
    ["Market timeline", "Clustering", "Prediction model", "Try the model", "How it works"]
)

unseen = [r for r in REGIMES if r not in set(res["train"]["true_regime"])]
if unseen:
    st.warning(
        f"In this dataset the training period (oldest 80% of dates) contains no **{', '.join(unseen)}** days, "
        "so the model has never seen that regime and can't predict it. Test accuracy drops for that reason, "
        "not because the method broke. It's a real limitation of learning from history: a model only knows "
        "the regimes it has seen."
    )

# ---------------------------------------------------------------- timeline
with tab_timeline:
    view = st.radio(
        "Colour the background by", ["True regime", "K-Means cluster"], horizontal=True,
        help="True regimes are hidden from K-Means; they're only used to check it.",
    )
    column, scale, title = (
        ("true_regime", REGIME_SCALE, "Regime") if view == "True regime"
        else ("cluster_name", CLUSTER_SCALE, "Cluster")
    )
    bands = periods(data, column)
    base = alt.Chart(data).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")))
    band_layer = alt.Chart(bands).mark_rect(opacity=0.28).encode(
        x="start:T", x2="end:T", color=alt.Color("label:N", scale=scale, title=title,
                                                  legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("label:N", title=title), alt.Tooltip("start:T", title="From"),
                 alt.Tooltip("end:T", title="To"), alt.Tooltip("days:Q", title="Trading days")],
    )
    price = base.mark_line(color="#1f2937", strokeWidth=1.3).encode(
        y=alt.Y("close:Q", title="Price", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("close:Q", title="Price", format=".2f"),
                 alt.Tooltip("true_regime:N", title="True regime"), alt.Tooltip("cluster_name:N", title="Cluster")],
    )
    split = alt.Chart(pd.DataFrame({"date": [test_start], "label": ["Test period →"]}))
    rule = split.mark_rule(strokeDash=[5, 4], color="#6b7280").encode(x="date:T")
    rule_text = split.mark_text(align="left", dx=6, dy=-6, color="#6b7280", fontSize=11).encode(
        x="date:T", y=alt.value(12), text="label:N")
    st.altair_chart((band_layer + price + rule + rule_text).properties(height=380), width="stretch")

    vol = base.mark_area(opacity=0.5, color="#6b7280", line={"color": "#374151", "strokeWidth": 1}).encode(
        y=alt.Y("rolling_volatility_20:Q", title="20-day volatility", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("rolling_volatility_20:Q", title="Volatility", format=".2%")],
    )
    st.altair_chart((vol + rule).properties(height=150), width="stretch")
    st.caption(
        "Volatility is the clearest signal: Volatile periods stand out immediately, while Bull, Bear and "
        "Stable differ mostly in their slow drift, which is much harder to see day to day."
    )

# ---------------------------------------------------------------- clustering
with tab_cluster:
    st.subheader("What K-Means found (without seeing any labels)")
    st.markdown(
        "K-Means grouped the days using four slow-moving features: 60-day trend, 20-day volatility, "
        "20-day momentum and relative volume. Each cluster is named from its own statistics, "
        "not from the true labels."
    )
    shown = profile.copy()
    table = pd.DataFrame({
        "Cluster": [cluster_names[c] for c in shown.index],
        "Days": shown["days"].astype(int).values,
        "60-day mean return": (shown["rolling_mean_60"] * 100).map("{:+.3f}%".format).values,
        "20-day volatility": (shown["rolling_volatility_20"] * 100).map("{:.2f}%".format).values,
        "20-day momentum": (shown["momentum_20"] * 100).map("{:+.1f}%".format).values,
        "Relative volume": shown["relative_volume_20"].map("{:.2f}×".format).values,
    })
    st.dataframe(table, hide_index=True, width="stretch")

    left, right = st.columns([3, 2])
    with left:
        ct = res["crosstab"].reindex(columns=REGIMES, fill_value=0)
        ct.index = [cluster_names[c] for c in ct.index]
        share = ct.div(ct.sum(axis=0), axis=1)
        long = ct.stack().rename("days").reset_index()
        long.columns = ["cluster", "true_regime", "days"]
        long["share"] = share.stack().values
        heat = alt.Chart(long).encode(
            x=alt.X("true_regime:N", sort=REGIMES, title="True regime", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("cluster:N", sort=cluster_domain, title=None, axis=alt.Axis(labelLimit=260)),
        )
        cells = heat.mark_rect().encode(
            color=alt.Color("share:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=[alt.Tooltip("cluster:N"), alt.Tooltip("true_regime:N", title="True regime"),
                     alt.Tooltip("days:Q"), alt.Tooltip("share:Q", title="Share of this regime", format=".0%")],
        )
        labels = heat.mark_text(fontSize=13).encode(
            text="days:Q",
            color=alt.condition(alt.datum.share > 0.5, alt.value("white"), alt.value("#111827")),
        )
        st.markdown("**Where each true regime's days ended up**")
        st.altair_chart((cells + labels).properties(height=240), width="stretch")
    with right:
        st.metric("Adjusted Rand index", f"{res['ari']:.2f}",
                  help="1 = clusters match the true regimes exactly, 0 = no better than random.")
        lines = []
        for regime in REGIMES:
            best = share[regime].idxmax()
            lines.append(f"- **{regime}**: {share.loc[best, regime]:.0%} of its days are in *{best}*")
        st.markdown(
            "\n".join(lines) + "\n\n"
            "Regimes that end up in the same cluster can't be told apart by K-Means. Bull and Stable "
            "usually do, because they differ mainly by a tiny drift (about 0.07% vs 0.01% a day) that "
            "distance-based clustering barely notices. The supervised model below separates them."
        )

# ---------------------------------------------------------------- model
with tab_model:
    st.subheader("Random Forest, tested on the future")
    st.markdown(
        f"Trained on **{res['train']['date'].min():%b %Y} – {res['train']['date'].max():%b %Y}** "
        f"({len(res['train']):,} days) and tested on **{test_start:%b %Y} – {test['date'].max():%b %Y}** "
        f"({len(test):,} days). Splitting by date matters: a random shuffle would put almost identical "
        "neighbouring days in both sets and overstate accuracy."
    )
    left, right = st.columns(2)
    with left:
        cm = metrics["confusion_matrix"].copy()
        cm.index = [i.replace("true ", "") for i in cm.index]
        cm.columns = [c.replace("pred ", "") for c in cm.columns]
        row_share = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
        cm_long = cm.stack().rename("days").reset_index()
        cm_long.columns = ["true", "predicted", "days"]
        cm_long["share"] = row_share.stack().values
        cm_base = alt.Chart(cm_long).encode(
            x=alt.X("predicted:N", sort=REGIMES, title="Predicted", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("true:N", sort=REGIMES, title="True"),
        )
        cm_chart = cm_base.mark_rect().encode(
            color=alt.Color("share:Q", scale=alt.Scale(scheme="greens"), legend=None),
            tooltip=["true:N", "predicted:N", "days:Q", alt.Tooltip("share:Q", format=".0%")],
        ) + cm_base.mark_text(fontSize=14).encode(
            text="days:Q", color=alt.condition(alt.datum.share > 0.5, alt.value("white"), alt.value("#111827")),
        )
        st.markdown("**Confusion matrix (test period)**")
        st.altair_chart(cm_chart.properties(height=260), width="stretch")
    with right:
        imp = res["importances"].rename(index=FEATURE_LABELS).rename("importance").reset_index()
        imp.columns = ["feature", "importance"]
        st.markdown("**What the model relies on**")
        st.altair_chart(
            alt.Chart(imp).mark_bar(color="#4f6d9a").encode(
                x=alt.X("importance:Q", title="Importance", axis=alt.Axis(format="%")),
                y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                tooltip=["feature:N", alt.Tooltip("importance:Q", format=".1%")],
            ).properties(height=260),
            width="stretch",
        )

    report = metrics["report"].loc[REGIMES, ["precision", "recall", "f1-score", "support"]].copy()
    report["support"] = report["support"].astype(int)
    report.columns = ["Precision", "Recall", "F1", "Test days"]
    left, right = st.columns(2)
    with left:
        st.markdown("**Per-regime scores (test period)**")
        st.dataframe(report.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}"}),
                     width="stretch")
    with right:
        st.markdown("**Cross-validation across five time blocks**")
        folds = cv.rename(columns={"fold": "Fold", "start": "Starts", "regimes_in_fold": "Regimes present",
                                   "accuracy": "Accuracy", "macro_f1": "Macro F1"})
        st.dataframe(folds.style.format({"Accuracy": "{:.1%}", "Macro F1": "{:.2f}"}),
                     hide_index=True, width="stretch")

    strip = pd.concat([
        periods(test, "true_regime").assign(series="True", row=0.0),
        periods(test, "predicted_regime").assign(series="Predicted", row=1.0),
    ])
    # Stretch each band to the next trading day so consecutive bands touch.
    strip["end"] = strip["end"] + pd.offsets.BDay(1)
    strip["row_end"] = strip["row"] + 0.8
    st.markdown("**Test period, day by day** (top: true regime, bottom: model's prediction)")
    st.altair_chart(
        alt.Chart(strip).mark_rect().encode(
            x=alt.X("start:T", title=None, axis=alt.Axis(format="%b %Y")), x2="end:T",
            y=alt.Y("row:Q", title=None, scale=alt.Scale(domain=[0, 1.8], reverse=True),
                    axis=alt.Axis(values=[0.4, 1.4], grid=False, ticks=False, domain=False,
                                  labelExpr="datum.value < 1 ? 'True' : 'Predicted'")),
            y2="row_end:Q",
            color=alt.Color("label:N", scale=REGIME_SCALE, title="Regime", legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("series:N", title="Row"), alt.Tooltip("label:N", title="Regime"),
                     alt.Tooltip("start:T", title="From"), alt.Tooltip("days:Q", title="Trading days")],
        ),
        # Height goes to Streamlit, not .properties(): there it would include the legend and axes.
        width="stretch", height=190,
    )
    st.caption("Most mistakes happen in the first weeks after a regime changes, while the rolling windows "
               "still contain the previous regime. Check the confusion matrix to see which regimes get mixed up.")

# ---------------------------------------------------------------- try it
with tab_try:
    st.subheader("Describe a market and let the model classify it")
    train = res["train"]
    preset = st.selectbox(
        "Start from a typical day of…", [r for r in REGIMES if r not in unseen],
        help="Fills the sliders with the median values of that regime in the training data.",
    )
    start = train[train["true_regime"] == preset][FEATURES].median()
    values = {}
    cols = st.columns(2)
    for i, feat in enumerate(FEATURES):
        scale, unit, fmt = DISPLAY[feat]
        lo, hi = train[feat].quantile([0.005, 0.995]) * scale
        values[feat] = cols[i % 2].slider(
            f"{FEATURE_LABELS[feat]} ({unit})", float(lo), float(hi),
            float(min(max(start[feat] * scale, lo), hi)), format=fmt,
            help=FEATURE_HELP[feat], key=f"{feat}-{preset}-{seed}",
        ) / scale

    model = res["model"]
    proba = pd.Series(model.predict_proba(pd.DataFrame([values])[FEATURES])[0], index=model.classes_)
    proba = proba.reindex(REGIMES).fillna(0)
    top = proba.idxmax()
    left, right = st.columns([1, 2])
    left.metric("Predicted regime", top, f"{proba[top]:.0%} confidence", delta_color="off", delta_arrow="off")
    right.altair_chart(
        alt.Chart(proba.rename("probability").rename_axis("regime").reset_index()).mark_bar().encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="Probability"),
            y=alt.Y("regime:N", sort=REGIMES, title=None),
            color=alt.Color("regime:N", scale=REGIME_SCALE, legend=None),
            tooltip=["regime:N", alt.Tooltip("probability:Q", format=".0%")],
        ).properties(height=170),
        width="stretch",
    )
    st.caption("Try raising 20-day volatility above about 2%: the prediction flips to Volatile. "
               "Changing only the mean return moves it between Bull, Stable and Bear.")

# ---------------------------------------------------------------- how it works
with tab_how:
    st.subheader("How it works")
    st.markdown(
        """
1. **Synthetic data.** 2,500 trading days where the market switches between four regimes in blocks of
   60–220 days. Each regime has its own average return, volatility and trading volume. 25 volume values are
   deleted on purpose.
2. **Cleaning.** Sort by date, drop duplicate dates, forward-fill missing volume (never filling from the future).
3. **Features.** 20- and 60-day average return and volatility, 20-day momentum, and volume relative to its own
   history. All look only backwards in time.
4. **Clustering.** K-Means on standardised slow-moving features. True labels are used only afterwards to score it.
5. **Prediction.** A Random Forest trained on the oldest 80% of dates and tested on the newest 20%, plus
   purged time-series cross-validation.

**Limitations.** The data is synthetic, so the regimes are cleaner than in real markets. Rolling windows mean
the model reacts to a regime change with a delay of a few weeks. Real regime labels don't exist, so on real data
only the unsupervised part applies directly.

*Educational project. Not investment advice.*
        """
    )
