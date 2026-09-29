"""Run the whole workflow in one call: generate -> clean -> features -> cluster -> predict.

Used by the web page, the notebook and the deploy build step, so they all show
exactly the same numbers. Running this file precomputes the results for the
default seed so the web page can load them instantly instead of retraining.
"""
import joblib

from clustering import compare_with_truth, describe_clusters, fit_clusters
from config import MODELS_DIR, RANDOM_STATE, REGIMES
from data_preprocessing import clean
from feature_engineering import add_features
from generate_data import generate
from prediction import train_and_evaluate


def run_pipeline(seed=RANDOM_STATE):
    raw = generate(seed=seed)
    data = add_features(clean(raw))
    data, _ = fit_clusters(data, seed=seed)
    crosstab, ari = compare_with_truth(data)
    results = train_and_evaluate(data, seed=seed)
    return {
        "seed": seed,
        "raw_rows": len(raw),
        "missing_volume_filled": int(raw["volume"].isna().sum()),
        "data": data,
        "cluster_profile": describe_clusters(data),
        "crosstab": crosstab,
        "ari": ari,
        **results,
    }


def cache_path(seed=RANDOM_STATE):
    return MODELS_DIR / f"pipeline_seed{seed}.joblib"


def load_or_run(seed=RANDOM_STATE):
    """Load precomputed results if they exist, otherwise compute them."""
    path = cache_path(seed)
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:  # stale file from another library version: just recompute
            pass
    return run_pipeline(seed)


ROBUSTNESS_SEEDS = [1, 7, 42, 123, 2024]


def robustness(seeds=ROBUSTNESS_SEEDS):
    """Rerun the whole pipeline on several synthetic datasets to show the typical range of results."""
    rows = []
    for seed in seeds:
        r = run_pipeline(seed)
        rows.append({
            "seed": seed,
            "test_accuracy": round(float(r["test_metrics"]["accuracy"]), 6),
            "cv_accuracy": round(float(r["cv"]["accuracy"].mean()), 6),
            "kmeans_ari": round(float(r["ari"]), 6),
            "regimes_missing_from_training": sorted(set(REGIMES) - set(r["train"]["true_regime"])),
        })
    return rows


if __name__ == "__main__":
    import json

    out = run_pipeline()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, cache_path(), compress=3)
    print(f"Precomputed seed {out['seed']}: test accuracy {out['test_metrics']['accuracy']:.3f}, "
          f"CV accuracy {out['cv']['accuracy'].mean():.3f}, ARI {out['ari']:.3f} -> {cache_path().name}")

    rows = robustness()
    (MODELS_DIR / "robustness.json").write_text(json.dumps(rows, indent=2))
    print("Robustness across seeds:", ", ".join(f"{r['seed']}: {r['test_accuracy']:.3f}" for r in rows))
