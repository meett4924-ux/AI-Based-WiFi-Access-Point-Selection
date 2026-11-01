"""
train_model.py
==============
AI-Based Wi-Fi Access Point Selection Using Machine Learning

This file does three things:
  1. Generates a SYNTHETIC Wi-Fi dataset (simulated, not real measurements).
  2. Trains a Random Forest Classifier that predicts network quality
     (Poor / Fair / Good / Excellent) for a single Access Point.
  3. Saves the trained model + its evaluation metrics with Joblib.

You can run it directly:      python train_model.py
The Streamlit app (app.py) also calls ensure_model() automatically, so the
model is trained on first launch if models/wifi_model.pkl does not exist.
"""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Paths (relative to this file, so the project works from any folder)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATASET_PATH = DATA_DIR / "wifi_dataset.csv"          # synthetic TRAINING data
SAMPLE_INPUT_PATH = DATA_DIR / "sample_ap_input.csv"   # sample file for CSV upload
MODEL_PATH = MODEL_DIR / "wifi_model.pkl"

# --------------------------------------------------------------------------
# Constants shared with the dashboard
# --------------------------------------------------------------------------
FEATURES = ["rssi", "users", "bandwidth", "latency", "congestion"]
CLASS_NAMES = ["Poor", "Fair", "Good", "Excellent"]   # class 0, 1, 2, 3
CLASS_SCORES = [12.5, 37.5, 62.5, 87.5]               # centre of each class on a 0-100 scale
SCORE_THRESHOLDS = [35, 50, 65]                       # quality score -> class boundaries
RANDOM_SEED = 42

# Valid ranges used for normalisation (also used by the Simulation Lab sliders)
LIMITS = {
    "rssi": (-90, -30),        # dBm  (higher / closer to 0 is better)
    "users": (1, 40),          # connected users (lower is better)
    "bandwidth": (1, 100),     # Mbps (higher is better)
    "latency": (5, 200),       # ms   (lower is better)
    "congestion": (1, 100),    # %    (lower is better)
}

# Example values from the project brief (used as the default simulation)
SAMPLE_APS = {
    "AP1": {"rssi": -75, "users": 20, "bandwidth": 10, "latency": 120, "congestion": 80},
    "AP2": {"rssi": -48, "users": 8, "bandwidth": 50, "latency": 30, "congestion": 20},
    "AP3": {"rssi": -62, "users": 15, "bandwidth": 25, "latency": 70, "congestion": 50},
}


# --------------------------------------------------------------------------
# Quality scoring (used ONLY to create training labels and as a tie-breaker)
# --------------------------------------------------------------------------
def normalized_components(df):
    """Convert each feature to a 0-1 value where 1 is always 'best'."""
    out = pd.DataFrame(index=df.index)
    for col in FEATURES:
        low, high = LIMITS[col]
        scaled = ((df[col].astype(float) - low) / (high - low)).clip(0, 1)
        # For users, latency and congestion a LOWER raw value is better
        out[col] = scaled if col in ("rssi", "bandwidth") else 1 - scaled
    return out


def quality_score(df):
    """Weighted network quality score from 0 (terrible) to 100 (excellent)."""
    n = normalized_components(df)
    score = (
        0.30 * n["rssi"]
        + 0.15 * n["users"]
        + 0.20 * n["bandwidth"]
        + 0.20 * n["latency"]
        + 0.15 * n["congestion"]
    )
    return score * 100


def score_to_class(scores):
    """Turn a 0-100 score into a class number (0=Poor ... 3=Excellent)."""
    return np.digitize(scores, SCORE_THRESHOLDS)


# --------------------------------------------------------------------------
# Synthetic dataset
# --------------------------------------------------------------------------
def generate_dataset(n_samples=6000, seed=RANDOM_SEED):
    """Create realistic-looking SIMULATED Wi-Fi measurements with labels."""
    rng = np.random.default_rng(seed)

    rssi = rng.uniform(-90, -30, n_samples)
    users = rng.integers(1, 41, n_samples)

    # More users usually means more congestion (plus some randomness)
    congestion = 0.7 * (users / 40 * 100) + 0.3 * rng.uniform(1, 100, n_samples)
    congestion = np.clip(congestion + rng.normal(0, 8, n_samples), 1, 100)

    # Bandwidth falls with weak signal and with congestion
    signal_factor = (rssi + 95) / 65
    bandwidth = 100 * signal_factor * (1 - 0.6 * congestion / 100) * rng.uniform(0.7, 1.15, n_samples)
    bandwidth = np.clip(bandwidth, 1, 100)

    # Latency rises with weak signal and with congestion
    latency = 10 + 0.9 * congestion + (-rssi - 30) * 0.8 + rng.normal(0, 10, n_samples)
    latency = np.clip(latency, 5, 200)

    df = pd.DataFrame(
        {
            "rssi": rssi.round(1),
            "users": users,
            "bandwidth": bandwidth.round(1),
            "latency": latency.round(1),
            "congestion": congestion.round(1),
        }
    )

    # Labels come from the scoring function plus a little noise, so the model
    # has to learn a pattern rather than copy an exact formula.
    true_score = quality_score(df)
    noisy_score = true_score + rng.normal(0, 2.5, n_samples)
    df["quality_score"] = true_score.round(2)
    df["quality_class"] = score_to_class(noisy_score)
    df["quality_label"] = df["quality_class"].map(dict(enumerate(CLASS_NAMES)))
    return df


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_and_save():
    """Generate data, train the Random Forest, evaluate it and save everything."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = generate_dataset()
    data.to_csv(DATASET_PATH, index=False)
    pd.DataFrame(
        [{"AP": ap, **vals} for ap, vals in SAMPLE_APS.items()]
    ).to_csv(SAMPLE_INPUT_PATH, index=False)

    X = data[FEATURES]
    y = data["quality_class"]

    # 80% training / 20% testing, same split every time (fixed seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Every metric below is calculated from real predictions on the test set
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    labels = list(model.classes_)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "labels": [CLASS_NAMES[i] for i in labels],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_counts": data["quality_label"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int).to_dict(),
        "feature_importance": dict(zip(FEATURES, model.feature_importances_.astype(float))),
    }

    bundle = {
        "model": model,
        "features": FEATURES,
        "metrics": metrics,
        "trained_on": "Synthetic (simulated) Wi-Fi data",
        "trained_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def ensure_model():
    """Load the saved model, or train a new one if it is missing / unreadable."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            if all(key in bundle for key in ("model", "features", "metrics")):
                return bundle
        except Exception:
            pass  # File is corrupted or from another scikit-learn version: retrain
    return train_and_save()


if __name__ == "__main__":
    result = train_and_save()
    m = result["metrics"]
    print("Model trained on synthetic data and saved to:", MODEL_PATH)
    print(f"Training samples : {m['n_train']}   Test samples: {m['n_test']}")
    print(f"Accuracy         : {m['accuracy']:.3f}")
    print(f"Precision        : {m['precision']:.3f}")
    print(f"Recall           : {m['recall']:.3f}")
    print(f"F1-score         : {m['f1']:.3f}")
    print("Class counts     :", m["class_counts"])
    print("Confusion matrix (rows=actual, cols=predicted):")
    for row in m["confusion_matrix"]:
        print("  ", row)
