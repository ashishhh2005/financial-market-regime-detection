"""Unsupervised regime detection with K-Means.

The true regime labels are never used to fit the clusters. They are only used
afterwards to measure how well the clusters line up with reality.
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE
from feature_engineering import CLUSTER_FEATURES

N_CLUSTERS = 4


def fit_clusters(df, n_clusters=N_CLUSTERS, seed=RANDOM_STATE):
    """Add a `cluster` column and return (df, fitted pipeline)."""
    model = make_pipeline(StandardScaler(), KMeans(n_clusters=n_clusters, random_state=seed, n_init=20))
    out = df.copy()
    out["cluster"] = model.fit_predict(out[CLUSTER_FEATURES])
    return out, model


def describe_clusters(df):
    """Profile each cluster and give it a plain-English name from its own statistics.

    Names come only from the features (volatility level and trend direction),
    not from the true labels.
    """
    profile = df.groupby("cluster")[CLUSTER_FEATURES].mean()
    profile["days"] = df.groupby("cluster").size()

    def name(row):
        vol = row["rolling_volatility_20"]  # typical daily move over the last 20 days
        level = "High volatility" if vol > 0.016 else "Moderate volatility" if vol > 0.008 else "Calm"
        drift = row["rolling_mean_60"]      # average daily return over the last 60 days
        direction = "flat" if abs(drift) < 0.0002 else "rising" if drift > 0 else "falling"
        return f"{level}, {direction}"

    profile["description"] = profile.apply(name, axis=1)
    return profile


def compare_with_truth(df):
    """Cross-tab of clusters vs true regimes, plus the adjusted Rand index (1 = perfect, 0 = random)."""
    table = pd.crosstab(df["cluster"], df["true_regime"])
    ari = adjusted_rand_score(df["true_regime"], df["cluster"])
    return table, ari


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from config import CLUSTER_COLORS, REGIME_COLORS, VIS_DIR
    from data_preprocessing import load_and_clean
    from feature_engineering import add_features

    VIS_DIR.mkdir(parents=True, exist_ok=True)
    data, _ = fit_clusters(add_features(load_and_clean()))
    profile = describe_clusters(data)
    table, ari = compare_with_truth(data)

    profile.round(5).to_csv(VIS_DIR / "cluster_summary.csv")
    pd.set_option("display.width", 160)
    print("Cluster profiles:\n", profile.round(4).to_string(), sep="")
    print("\nClusters vs true regimes (labels used only for this check):\n", table.to_string(), sep="")
    print(f"\nAdjusted Rand index: {ari:.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for regime, color in REGIME_COLORS.items():
        part = data[data["true_regime"] == regime]
        axes[0].scatter(part["date"], part["close"], s=5, color=color, label=regime)
    axes[0].set_title("True regimes (hidden from K-Means)")
    axes[0].set_ylabel("Price")
    axes[0].legend(markerscale=3, ncol=4, loc="upper left")

    cmap = CLUSTER_COLORS
    for c, row in profile.iterrows():
        part = data[data["cluster"] == c]
        axes[1].scatter(part["date"], part["close"], s=5, color=cmap[c], label=f"{c}: {row['description']}")
    axes[1].set_title(f"K-Means clusters (adjusted Rand index {ari:.2f})")
    axes[1].set_ylabel("Price")
    axes[1].set_xlabel("Date")
    axes[1].legend(markerscale=3, ncol=2, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(VIS_DIR / "market_regimes.png", dpi=130)
    print(f"\nSaved {VIS_DIR.name}/cluster_summary.csv and {VIS_DIR.name}/market_regimes.png")
