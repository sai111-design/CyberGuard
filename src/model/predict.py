"""Risk scorer — scores org controls using the trained anomaly model.

Uses NIST-anchored normalisation (Fix 1), floor-protected formula (Fix 2),
and threshold ceiling removal (Fix 3). Never refits the scaler on org data.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.model.train import load_model
from src.detection.risk_scorer import (
    apply_floor_formula,
    compute_gap_weight,
    compute_risk_level,
)


def score_controls(org_df: pd.DataFrame, model, nist_scaler) -> pd.DataFrame:
    """Score org controls and return DataFrame with risk columns added.

    Fix 1 — NIST-Anchored Normalisation:
      Uses nist_scaler.transform() only (NEVER .fit() or .fit_transform()).
      Clips to [0, 1] to handle org scores outside NIST training range.

    Fix 2 + Fix 3 — Floor-Protected Formula:
      Missing controls always ≥ 7.0, planned ≥ 4.0.
      Partial/planned can reach CRITICAL when anomaly_score is high enough.
    """
    result_df = org_df.copy()

    # Load original NIST descriptions for model scoring.
    # Missing controls have empty descriptions in org_controls (to test the floor),
    # but the model needs real text to produce meaningful anomaly scores.
    nist_path = os.path.join("data", "processed", "controls_clean.csv")
    nist_desc_map = pd.read_csv(nist_path).set_index("control_id")["description"].to_dict()

    # Use original NIST description for scoring; fall back to org description
    descriptions = [
        nist_desc_map.get(cid, desc) if not desc else desc
        for cid, desc in zip(result_df["control_id"], result_df["description"].fillna(""))
    ]

    # Fix 1: NIST-anchored normalisation
    raw_scores = model.decision_function(descriptions)
    anomaly_norm = (
        nist_scaler.transform((-raw_scores).reshape(-1, 1))
        .flatten()
        .clip(0, 1)
    )

    # Anomaly flag (from IsolationForest predictions: -1 = anomaly, 1 = normal)
    predictions = model.predict(descriptions)
    anomaly_flags = predictions == -1

    result_df["anomaly_flag"] = anomaly_flags
    result_df["anomaly_score"] = np.round(anomaly_norm, 6)

    # Fix 2 + Fix 3: Floor-protected formula per row
    result_df["gap_weight"] = result_df["status"].apply(compute_gap_weight)
    result_df["risk_score"] = result_df.apply(
        lambda row: apply_floor_formula(row["anomaly_score"], row["status"]),
        axis=1,
    )
    result_df["risk_level"] = result_df["risk_score"].apply(compute_risk_level)

    return result_df


def main():
    # Load model and scaler
    model, nist_scaler = load_model()
    print("✓ Loaded anomaly_model.pkl and nist_scaler.pkl")

    # Load org controls
    org_path = os.path.join("data", "processed", "org_controls.csv")
    org_df = pd.read_csv(org_path)
    print(f"✓ Loaded {len(org_df)} org controls from {org_path}")

    # Score
    risk_df = score_controls(org_df, model, nist_scaler)

    # Save
    out_path = os.path.join("data", "processed", "risk_register.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    risk_df.to_csv(out_path, index=False)
    print(f"\n✓ Wrote {len(risk_df)} rows to {out_path}")

    # ── Verification ───────────────────────────────────────────────────────────
    print(f"\nrisk_level value counts:")
    print(risk_df["risk_level"].value_counts().to_string())

    print(f"\nrisk_score range: [{risk_df['risk_score'].min()}, {risk_df['risk_score'].max()}]")

    missing_rows = risk_df[risk_df["status"] == "missing"]
    print(f"\nMissing-status rows: {len(missing_rows)}")
    print(f"  Min risk_score among missing: {missing_rows['risk_score'].min()}")

    anomaly_count = risk_df["anomaly_flag"].sum()
    print(f"\nAnomaly flags (True): {anomaly_count}")

    # Checkpoint assertions
    assert risk_df["risk_score"].min() >= 0, "risk_score below 0!"
    assert risk_df["risk_score"].max() <= 10, "risk_score above 10!"
    assert set(risk_df["risk_level"].unique()).issubset(
        {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    ), "Invalid risk_level values!"
    assert missing_rows["risk_score"].min() >= 7.0, (
        f"Missing control floor violated! Min = {missing_rows['risk_score'].min()}"
    )
    assert "CRITICAL" in risk_df["risk_level"].values, "No CRITICAL rows!"
    assert "HIGH" in risk_df["risk_level"].values, "No HIGH rows!"
    print("\n✓ All checkpoint assertions passed")


if __name__ == "__main__":
    main()
