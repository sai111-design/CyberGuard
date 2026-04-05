"""TF-IDF + IsolationForest trainer.

Trains an anomaly detection pipeline on NIST control descriptions and saves
both the model and a NIST-anchored MinMaxScaler as separate artifacts.
The scaler is NEVER refitted on org data — only .transform() at inference time.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# ── GRC stop words (added on top of sklearn "english") ─────────────────────────
GRC_STOP_WORDS = [
    "shall", "must", "should", "may", "organization", "system", "information",
    "security", "control", "policy", "procedure", "ensure", "provide", "implement",
    "establish", "maintain", "review", "define", "document", "required", "applicable",
]

# Combine with sklearn's built-in english stop words
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
COMBINED_STOP_WORDS = list(ENGLISH_STOP_WORDS) + GRC_STOP_WORDS

# ── Artifact paths ─────────────────────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join("src", "model", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "anomaly_model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "nist_scaler.pkl")


def train_model():
    """Train TF-IDF + IsolationForest pipeline on NIST controls and save artifacts."""

    # 1. Load training corpus
    controls_path = os.path.join("data", "processed", "controls_clean.csv")
    controls_df = pd.read_csv(controls_path)
    corpus = controls_df["description"].fillna("").tolist()
    print(f"Training corpus: {len(corpus)} control descriptions")

    # 2. Build pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words=COMBINED_STOP_WORDS,
            sublinear_tf=True,
            max_df=0.85,
            min_df=2,
        )),
        ("isoforest", IsolationForest(
            n_estimators=200,
            contamination=0.1,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # 3. Fit on NIST descriptions
    model.fit(corpus)
    print("✓ Pipeline fitted (TF-IDF → IsolationForest)")

    # 4. Compute decision scores on training data
    nist_raw_scores = model.decision_function(corpus)

    # 5. Fit MinMaxScaler on NIST scores ONLY (negated so higher = more anomalous)
    nist_scaler = MinMaxScaler()
    nist_scaler.fit((-nist_raw_scores).reshape(-1, 1))
    print(f"  Scaler fitted on NIST scores — range: [{nist_raw_scores.min():.4f}, {nist_raw_scores.max():.4f}]")

    # 6–7. Save both artifacts
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved model  → {MODEL_PATH}")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(nist_scaler, f)
    print(f"  Saved scaler → {SCALER_PATH}")

    print(f"\n✓ Training complete. {len(corpus)} controls, 2 artifacts saved.")


def load_model():
    """Load and return (model, nist_scaler). Raises FileNotFoundError if missing."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train.py first.")
    if not os.path.isfile(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}. Run train.py first.")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        nist_scaler = pickle.load(f)

    return model, nist_scaler


if __name__ == "__main__":
    train_model()
