"""Pytest suite — 7 tests validating the full CyberGuard pipeline outputs."""

import os

import pandas as pd
import pytest

DATA_DIR = os.path.join("data", "processed")
ARTIFACTS_DIR = os.path.join("src", "model", "artifacts")


def test_controls_clean_exists():
    """controls_clean.csv exists with correct columns and no null control_ids."""
    path = os.path.join(DATA_DIR, "controls_clean.csv")
    assert os.path.isfile(path), f"{path} does not exist"

    df = pd.read_csv(path)
    expected_cols = [
        "control_id", "family_code", "control_family",
        "nist_csf_function", "nist_csf_category",
        "description", "source_row_id",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"

    assert df["control_id"].isna().sum() == 0, "Null control_id values found"


def test_org_controls_status_values():
    """All status values in org_controls.csv are valid."""
    path = os.path.join(DATA_DIR, "org_controls.csv")
    df = pd.read_csv(path)
    valid = {"implemented", "partial", "planned", "missing"}
    actual = set(df["status"].unique())
    assert actual.issubset(valid), f"Invalid status values: {actual - valid}"


def test_org_controls_missing_descriptions():
    """Every row with status=='missing' has empty or NaN description."""
    path = os.path.join(DATA_DIR, "org_controls.csv")
    df = pd.read_csv(path)
    missing_rows = df[df["status"] == "missing"]
    for idx, row in missing_rows.iterrows():
        desc = row["description"]
        assert pd.isna(desc) or desc == "", (
            f"Row {idx} (control_id={row['control_id']}): "
            f"status='missing' but description='{desc}'"
        )


def test_risk_register_score_range():
    """risk_score column is between 0 and 10 inclusive for all rows."""
    path = os.path.join(DATA_DIR, "risk_register.csv")
    df = pd.read_csv(path)
    assert df["risk_score"].min() >= 0, f"risk_score below 0: {df['risk_score'].min()}"
    assert df["risk_score"].max() <= 10, f"risk_score above 10: {df['risk_score'].max()}"


def test_missing_controls_floor():
    """Every risk_register row with status='missing' has risk_score >= 7.0."""
    path = os.path.join(DATA_DIR, "risk_register.csv")
    df = pd.read_csv(path)
    missing_rows = df[df["status"] == "missing"]
    assert len(missing_rows) > 0, "No missing-status rows found"
    violators = missing_rows[missing_rows["risk_score"] < 7.0]
    assert len(violators) == 0, (
        f"{len(violators)} missing-status rows have risk_score < 7.0:\n"
        f"{violators[['control_id', 'risk_score']].to_string()}"
    )


def test_alerts_risk_levels():
    """alerts.csv risk_level column only contains CRITICAL or HIGH."""
    path = os.path.join(DATA_DIR, "alerts.csv")
    df = pd.read_csv(path)
    valid = {"CRITICAL", "HIGH"}
    actual = set(df["risk_level"].unique())
    assert actual.issubset(valid), f"Invalid alert risk_levels: {actual - valid}"


def test_model_artifacts_exist():
    """Both anomaly_model.pkl and nist_scaler.pkl exist."""
    model_path = os.path.join(ARTIFACTS_DIR, "anomaly_model.pkl")
    scaler_path = os.path.join(ARTIFACTS_DIR, "nist_scaler.pkl")
    assert os.path.isfile(model_path), f"{model_path} does not exist"
    assert os.path.isfile(scaler_path), f"{scaler_path} does not exist"
