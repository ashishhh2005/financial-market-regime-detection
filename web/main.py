"""Market Regime Detection - web page and JSON API (FastAPI).

Run locally:   uvicorn web.main:app --reload
Then open:     http://127.0.0.1:8000        (web page)
               http://127.0.0.1:8000/docs   (interactive API docs)

The page is plain HTML rendered on the server, so it displays fully even if
JavaScript is blocked. JavaScript only makes the prediction form update
without reloading the page.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import charts  # noqa: E402
from config import CLUSTER_COLORS, MODELS_DIR, RANDOM_STATE, REGIME_COLORS, REGIMES  # noqa: E402
from feature_engineering import FEATURE_LABELS, FEATURES  # noqa: E402
from pipeline import load_or_run  # noqa: E402

WEB_DIR = Path(__file__).resolve().parent
REPO_URL = os.environ.get("REPO_URL", "https://github.com/ashishhh2005/financial-market-regime-detection")

# ------------------------------------------------------------------ load results once at startup
RES = load_or_run(RANDOM_STATE)
if not (charts.CHART_DIR / "timeline.svg").exists():
    charts.build_all(RES)
ROBUSTNESS_PATH = MODELS_DIR / "robustness.json"
ROBUSTNESS = json.loads(ROBUSTNESS_PATH.read_text()) if ROBUSTNESS_PATH.exists() else []

MODEL = RES["model"]
TRAIN, TEST = RES["train"], RES["test"]
KNOWN_REGIMES = [r for r in REGIMES if r in set(TRAIN["true_regime"])]

# How each feature appears in the form: shown value = raw value * scale.
DISPLAY = {
    "rolling_mean_20": (100, "%", 0.001, 3),
    "rolling_volatility_20": (100, "%", 0.01, 2),
    "rolling_mean_60": (100, "%", 0.001, 3),
    "rolling_volatility_60": (100, "%", 0.01, 2),
    "momentum_20": (100, "%", 0.1, 1),
    "relative_volume_20": (1, "×", 0.01, 2),
}
FEATURE_HELP = {
    "rolling_mean_20": "Average daily return over the last 20 trading days.",
    "rolling_volatility_20": "How much daily returns swung over the last 20 trading days.",
    "rolling_mean_60": "Average daily return over the last 60 trading days.",
    "rolling_volatility_60": "How much daily returns swung over the last 60 trading days.",
    "momentum_20": "Price change over the last 20 trading days.",
    "relative_volume_20": "Recent trading volume compared with normal (1.00 = normal).",
}
LOW, HIGH = TRAIN[FEATURES].quantile(0.005), TRAIN[FEATURES].quantile(0.995)
PRESETS = {r: TRAIN.loc[TRAIN["true_regime"] == r, FEATURES].median() for r in KNOWN_REGIMES}


def predict(values):
    """values: raw feature values -> (predicted regime, {regime: probability})."""
    row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    proba = pd.Series(MODEL.predict_proba(row)[0], index=MODEL.classes_).reindex(REGIMES).fillna(0.0)
    return proba.idxmax(), {r: float(p) for r, p in proba.items()}


def clamp(feature, raw):
    return float(min(max(raw, LOW[feature]), HIGH[feature]))


def pct(x, digits=1):
    return f"{x * 100:.{digits}f}%"


# ------------------------------------------------------------------ page content that never changes
def build_static_context():
    metrics, cv, profile = RES["test_metrics"], RES["cv"], RES["cluster_profile"]
    names = {c: f"{c}: {d}" for c, d in profile["description"].items()}

    ct = RES["crosstab"].reindex(columns=REGIMES, fill_value=0)
    share = ct.div(ct.sum(axis=0).replace(0, 1), axis=1)
    cluster_findings = []
    for regime in REGIMES:
        if ct[regime].sum() == 0:
            continue
        best = share[regime].idxmax()
        cluster_findings.append({"regime": regime, "share": pct(share.loc[best, regime], 0),
                                 "cluster": names[best], "color": REGIME_COLORS[regime]})

    report = metrics["report"]
    return {
        "repo_url": REPO_URL,
        "kpis": [
            {"label": "Test accuracy", "value": pct(metrics["accuracy"]),
             "note": f"vs {pct(RES['baseline_accuracy'], 0)} baseline", "good": True,
             "help": f"Newest 20% of dates ({len(TEST)} days), never seen in training."},
            {"label": "Cross-validated accuracy", "value": pct(cv["accuracy"].mean()),
             "note": f"range {pct(cv['accuracy'].min(), 0)} to {pct(cv['accuracy'].max(), 0)}",
             "help": "Five consecutive time blocks, each tested once, with a 60-day gap around it."},
            {"label": "Cluster agreement", "value": f"{RES['ari']:.2f}",
             "note": "adjusted Rand index", "help": "How well K-Means matches the hidden regimes: 1 = perfect, 0 = random."},
            {"label": "Trading days", "value": f"{len(RES['data']):,}",
             "note": f"{len(charts.periods(RES['data'], 'true_regime'))} regime periods",
             "help": f"{RES['missing_volume_filled']} missing volume values were filled during cleaning."},
        ],
        "train_range": f"{TRAIN['date'].min():%b %Y} – {TRAIN['date'].max():%b %Y}",
        "test_range": f"{TEST['date'].min():%b %Y} – {TEST['date'].max():%b %Y}",
        "n_train": f"{len(TRAIN):,}", "n_test": f"{len(TEST):,}",
        "clusters": [
            {"name": names[c], "color": CLUSTER_COLORS[c], "days": f"{int(row['days']):,}",
             "trend": f"{row['rolling_mean_60'] * 100:+.3f}%", "vol": f"{row['rolling_volatility_20'] * 100:.2f}%",
             "momentum": f"{row['momentum_20'] * 100:+.1f}%", "volume": f"{row['relative_volume_20']:.2f}×"}
            for c, row in profile.iterrows()
        ],
        "cluster_findings": cluster_findings,
        "ari": f"{RES['ari']:.2f}",
        "report": [
            {"regime": r, "color": REGIME_COLORS[r], "precision": f"{report.loc[r, 'precision']:.2f}",
             "recall": f"{report.loc[r, 'recall']:.2f}", "f1": f"{report.loc[r, 'f1-score']:.2f}",
             "days": int(report.loc[r, "support"])}
            for r in REGIMES
        ],
        "cv": [
            {"fold": int(row.fold), "start": str(row.start), "regimes": int(row.regimes_in_fold),
             "accuracy": pct(row.accuracy), "f1": f"{row.macro_f1:.2f}"}
            for row in cv.itertuples()
        ],
        "robustness": [
            {"seed": r["seed"], "test": pct(r["test_accuracy"]), "cv": pct(r["cv_accuracy"]),
             "ari": f"{r['kmeans_ari']:.2f}", "missing": ", ".join(r["regimes_missing_from_training"]) or "–",
             "is_default": r["seed"] == RANDOM_STATE}
            for r in ROBUSTNESS
        ],
        "robust_cv_range": (f"{pct(min(r['cv_accuracy'] for r in ROBUSTNESS), 0)} to "
                            f"{pct(max(r['cv_accuracy'] for r in ROBUSTNESS), 0)}") if ROBUSTNESS else "",
        "regimes": [{"name": r, "color": REGIME_COLORS[r]} for r in REGIMES],
    }


def chart_sizes():
    """Width/height of each SVG, so the page can reserve space before images load (no layout jumps)."""
    import re
    sizes = {}
    for svg in charts.CHART_DIR.glob("*.svg"):
        head = svg.read_text()[:600]
        w, h = (re.search(rf'{k}="([\d.]+)pt"', head) for k in ("width", "height"))
        if w and h:
            sizes[svg.stem] = {"w": round(float(w.group(1))), "h": round(float(h.group(1)))}
    return sizes


STATIC_CONTEXT = {**build_static_context(), "chart": chart_sizes()}

# ------------------------------------------------------------------ app
app = FastAPI(
    title="Market Regime Detection API",
    description="Predict the market regime (Bull, Bear, Stable, Volatile) from six backward-looking market "
                "features. Trained on synthetic data - educational project, not investment advice.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=WEB_DIR / "templates")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home(request: Request):
    params = request.query_params
    preset = params.get("preset") if params.get("preset") in PRESETS else KNOWN_REGIMES[0]
    values = {}
    for f in FEATURES:
        scale = DISPLAY[f][0]
        try:
            values[f] = clamp(f, float(params[f]) / scale) if f in params else float(PRESETS[preset][f])
        except ValueError:
            values[f] = float(PRESETS[preset][f])
    regime, proba = predict(values)

    sliders = []
    for f in FEATURES:
        scale, unit, step, digits = DISPLAY[f]
        sliders.append({
            "name": f, "label": FEATURE_LABELS[f], "unit": unit, "help": FEATURE_HELP[f],
            "scale": scale, "step": step, "digits": digits,
            "min": round(LOW[f] * scale, digits), "max": round(HIGH[f] * scale, digits),
            "value": round(values[f] * scale, digits),
        })
    context = {
        **STATIC_CONTEXT,
        "presets": KNOWN_REGIMES, "preset": preset, "sliders": sliders,
        "prediction": {"regime": regime, "color": REGIME_COLORS[regime], "confidence": pct(proba[regime], 0),
                       "bars": [{"regime": r, "color": REGIME_COLORS[r], "pct": round(p * 100, 1)}
                                for r, p in proba.items()]},
        "submitted": any(f in params for f in FEATURES),
    }
    return templates.TemplateResponse(request, "index.html", context)


class MarketFeatures(BaseModel):
    """Raw feature values (fractions, not percentages)."""
    rolling_mean_20: float = Field(..., description="Average daily return over the last 20 days, e.g. 0.0006")
    rolling_volatility_20: float = Field(..., ge=0, description="Std-dev of daily returns over 20 days, e.g. 0.008")
    rolling_mean_60: float = Field(..., description="Average daily return over the last 60 days")
    rolling_volatility_60: float = Field(..., ge=0, description="Std-dev of daily returns over 60 days")
    momentum_20: float = Field(..., description="Price change over the last 20 days, e.g. 0.012 for +1.2%")
    relative_volume_20: float = Field(..., gt=0, description="20-day average volume / normal volume, e.g. 1.0")

    model_config = {"json_schema_extra": {"examples": [
        {k: round(float(v), 6) for k, v in PRESETS[KNOWN_REGIMES[0]].items()}
    ]}}


class Prediction(BaseModel):
    regime: str
    probabilities: dict[str, float]


@app.post("/api/predict", response_model=Prediction, tags=["model"])
def api_predict(features: MarketFeatures):
    """Classify one market observation into a regime."""
    regime, proba = predict(features.model_dump())
    return {"regime": regime, "probabilities": {r: round(p, 4) for r, p in proba.items()}}


@app.get("/api/summary", tags=["model"])
def api_summary():
    """Headline evaluation results for the default dataset."""
    return {
        "test_accuracy": round(float(RES["test_metrics"]["accuracy"]), 4),
        "test_macro_f1": round(float(RES["test_metrics"]["macro_f1"]), 4),
        "baseline_accuracy": round(float(RES["baseline_accuracy"]), 4),
        "cv_accuracy_mean": round(float(RES["cv"]["accuracy"].mean()), 4),
        "kmeans_adjusted_rand_index": round(float(RES["ari"]), 4),
        "train_days": len(TRAIN), "test_days": len(TEST),
        "features": FEATURES, "regimes": REGIMES,
        "robustness": ROBUSTNESS,
    }


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}
