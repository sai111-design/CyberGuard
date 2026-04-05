"""Risk scorer helper module — gap weight, floor formula, and risk level classification.

This is a HELPER MODULE ONLY — no main() block, no CSV reading.
Four pure functions implementing the revised floor-protected risk formula.
"""

# ── Status-based constants ─────────────────────────────────────────────────────
STATUS_WEIGHT = {
    "implemented": 0.1,
    "partial": 0.5,
    "planned": 0.7,
    "missing": 1.0,
}

STATUS_FLOOR = {
    "implemented": 0.0,
    "partial": 2.0,
    "planned": 4.0,
    "missing": 7.0,
}


def compute_gap_weight(status: str) -> float:
    """Returns the multiplier for this control's implementation status."""
    return STATUS_WEIGHT.get(status.lower(), 1.0)


def compute_base_floor(status: str) -> float:
    """Returns the minimum risk score for this status regardless of ML score.

    Fixes the missing-data zero trap: a missing control with blank description
    must never score below 7.0. A planned control must never score below 4.0.
    """
    return STATUS_FLOOR.get(status.lower(), 7.0)


def compute_risk_level(risk_score: float) -> str:
    """Maps 0–10 risk score to CRITICAL/HIGH/MEDIUM/LOW. Always returns uppercase."""
    if risk_score >= 8.0:
        return "CRITICAL"
    if risk_score >= 6.0:
        return "HIGH"
    if risk_score >= 4.0:
        return "MEDIUM"
    return "LOW"


def apply_floor_formula(anomaly_score: float, status: str) -> float:
    """Floor-protected risk formula with anomaly-aware missing-control boost.

    For missing-status controls:
      base = 7.0, boost = anomaly_score * 3.0 → range [7.0, 10.0]
      This ensures every missing control scores differently based on how
      anomalous its context is, making the model's pattern learning visible.

    For all other statuses:
      ml_score   = anomaly_score * gap_weight * 10
      risk_score = max(base_floor, ml_score)  ← ML can only push UP from floor
    """
    if status.lower() == "missing":
        base = 7.0
        boost = anomaly_score * 3.0
        return round(min(base + boost, 10.0), 2)

    ml_score = anomaly_score * compute_gap_weight(status) * 10
    base_floor = compute_base_floor(status)
    return round(min(max(base_floor, ml_score), 10.0), 2)
