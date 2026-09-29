"""Supervised regime prediction with a Random Forest, evaluated on unseen future dates."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE, REGIMES, TRAIN_FRACTION
from evaluation import evaluate, purged_cv_scores, time_split
from feature_engineering import FEATURES


def build_model(seed=RANDOM_STATE):
    # Tree models don't need feature scaling, so there's no scaler here.
    return RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def train_and_evaluate(df, seed=RANDOM_STATE):
    """Train on the oldest 80% of days, test on the newest 20%, and run purged CV."""
    train, test = time_split(df, TRAIN_FRACTION)
    model = build_model(seed).fit(train[FEATURES], train["true_regime"])
    pred = model.predict(test[FEATURES])
    return {
        "model": model,
        "train": train,
        "test": test.assign(predicted_regime=pred),
        "test_metrics": evaluate(test["true_regime"], pred, REGIMES),
        "cv": purged_cv_scores(build_model(seed), df[FEATURES], df["true_regime"], dates=df["date"]),
        "baseline_accuracy": (test["true_regime"] == train["true_regime"].mode()[0]).mean(),
        "importances": pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False),
    }


if __name__ == "__main__":
    from config import MODEL_PATH
    from data_preprocessing import load_and_clean
    from evaluation import print_evaluation
    from feature_engineering import add_features

    results = train_and_evaluate(add_features(load_and_clean()))
    train, test = results["train"], results["test"]
    unseen = sorted(set(REGIMES) - set(train["true_regime"]))
    if unseen:
        print(f"WARNING: the training period has no {', '.join(unseen)} days, so the model can't predict them.\n")
    print(f"Train: {train['date'].min():%Y-%m-%d} to {train['date'].max():%Y-%m-%d} ({len(train)} days)")
    print(f"Test:  {test['date'].min():%Y-%m-%d} to {test['date'].max():%Y-%m-%d} ({len(test)} days)\n")
    print_evaluation(results["test_metrics"])
    print(f"\nBaseline (always predict the most common training regime): {results['baseline_accuracy']:.3f}")

    cv = results["cv"]
    print("\nPurged 5-fold time-series cross-validation:")
    print(cv.round(3).to_string(index=False))
    print(f"Mean accuracy {cv['accuracy'].mean():.3f} (range {cv['accuracy'].min():.3f}-{cv['accuracy'].max():.3f})")

    print("\nFeature importance:")
    print(results["importances"].round(3).to_string())

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": results["model"], "features": FEATURES}, MODEL_PATH, compress=3)
    print(f"\nSaved {MODEL_PATH.parent.name}/{MODEL_PATH.name}")
