"""Project-wide paths and settings.

Paths are resolved from this file's location, so every script works no matter
which folder you run it from (project root, src/, or notebooks/).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
VIS_DIR = ROOT / "visualizations"

DATA_PATH = DATA_DIR / "market_data.csv"
MODEL_PATH = MODELS_DIR / "regime_classifier.joblib"

RANDOM_STATE = 42
REGIMES = ["Bull", "Bear", "Stable", "Volatile"]
REGIME_COLORS = {
    "Bull": "#2a9d8f",
    "Bear": "#d1495b",
    "Stable": "#4f6d9a",
    "Volatile": "#e9a23b",
}
# Clusters get their own palette so they're never confused with the true regimes.
CLUSTER_COLORS = ["#7b61ff", "#1b9ad6", "#a0522d", "#6b7280", "#c026d3", "#0f766e"]

# Share of the timeline (oldest first) used for training. The newest 20% is the test set.
TRAIN_FRACTION = 0.80
