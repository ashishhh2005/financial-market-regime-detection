"""Evaluation helpers shared by the scripts, the notebook and the web app."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def time_split(df, train_fraction):
    """Split by date: oldest rows for training, newest rows for testing.

    A random shuffle would put near-identical neighbouring days (which share
    most of their rolling window) on both sides and inflate the score.
    """
    cut = int(len(df) * train_fraction)
    return df.iloc[:cut], df.iloc[cut:]


def purged_cv_scores(model, X, y, n_splits=5, gap=60, dates=None):
    """Blocked k-fold cross-validation for time series.

    Each fold is one contiguous stretch of time. Training rows within `gap`
    days of the fold are dropped so overlapping rolling windows can't leak.
    """
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    fold_dates = None if dates is None else pd.to_datetime(dates).reset_index(drop=True).dt.date
    folds = np.array_split(np.arange(len(X)), n_splits)
    rows = []
    for i, test_idx in enumerate(folds, start=1):
        lo, hi = test_idx.min(), test_idx.max()
        train_idx = np.setdiff1d(np.arange(len(X)), np.arange(max(0, lo - gap), hi + gap + 1))
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        y_test = y.iloc[test_idx]
        pred = fitted.predict(X.iloc[test_idx])
        # A fold covers one stretch of time and may not contain every regime,
        # so macro F1 is averaged over the regimes that actually occur in it.
        present = sorted(y_test.unique())
        rows.append({
            "fold": i,
            "start": fold_dates[lo] if fold_dates is not None else lo,
            "regimes_in_fold": len(present),
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, labels=present, average="macro", zero_division=0),
        })
    return pd.DataFrame(rows)


def evaluate(y_true, y_pred, labels):
    """Return accuracy, macro F1, a per-class report and a labelled confusion matrix."""
    report = pd.DataFrame(
        classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    ).T
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"true {l}" for l in labels],
        columns=[f"pred {l}" for l in labels],
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "report": report,
        "confusion_matrix": cm,
    }


def print_evaluation(result):
    print(f"Accuracy: {result['accuracy']:.3f}   Macro F1: {result['macro_f1']:.3f}")
    print("\nPer-class report:")
    print(result["report"].round(3).to_string())
    print("\nConfusion matrix:")
    print(result["confusion_matrix"].to_string())
