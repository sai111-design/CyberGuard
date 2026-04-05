"""Org inventory generator — simulates an organisation's control implementation status."""

import os
import random
from datetime import date, timedelta

import pandas as pd


# ── Owner mapping by family_code ───────────────────────────────────────────────
OWNER_MAP = {
    "AC": "Security Team",
    "IA": "Security Team",
    "AU": "Security Team",
    "CM": "IT Operations",
    "SA": "IT Operations",
    "SC": "IT Operations",
    "IR": "Incident Response Team",
    "CP": "Incident Response Team",
    "RA": "Risk & Compliance",
    "CA": "Risk & Compliance",
    "PM": "Risk & Compliance",
    "AT": "HR & Training",
    "PS": "HR & Training",
}

# ── Status weights ─────────────────────────────────────────────────────────────
STATUS_CHOICES = ["implemented", "partial", "planned", "missing"]
STATUS_WEIGHTS = [0.40, 0.25, 0.15, 0.20]

# ── Date range for last_reviewed ───────────────────────────────────────────────
DATE_START = date(2024, 1, 1)
DATE_END = date(2024, 12, 31)
DATE_RANGE_DAYS = (DATE_END - DATE_START).days  # 365


def main():
    random.seed(42)

    # Load controls_clean.csv
    controls_path = os.path.join("data", "processed", "controls_clean.csv")
    controls_df = pd.read_csv(controls_path)
    print(f"Loaded {len(controls_df)} controls from {controls_path}")

    rows = []
    for _, ctrl in controls_df.iterrows():
        cid = ctrl["control_id"]
        family_code = ctrl["family_code"]
        control_family = ctrl["control_family"]
        nist_csf_function = ctrl["nist_csf_function"]
        description = ctrl["description"]

        # Weighted random status
        status = random.choices(STATUS_CHOICES, weights=STATUS_WEIGHTS, k=1)[0]

        # Owner from family_code
        owner = OWNER_MAP.get(family_code, "Security Team")

        # Random date in 2024
        days_offset = random.randint(0, DATE_RANGE_DAYS)
        last_reviewed = (DATE_START + timedelta(days=days_offset)).strftime("%Y-%m-%d")

        # Missing-status controls get empty description (tests floor formula)
        if status == "missing":
            description = ""

        rows.append({
            "control_id": cid,
            "family_code": family_code,
            "control_family": control_family,
            "nist_csf_function": nist_csf_function,
            "description": description,
            "status": status,
            "owner": owner,
            "last_reviewed": last_reviewed,
        })

    # Write output
    col_order = [
        "control_id", "family_code", "control_family", "nist_csf_function",
        "description", "status", "owner", "last_reviewed",
    ]
    org_df = pd.DataFrame(rows)[col_order]

    out_path = os.path.join("data", "processed", "org_controls.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    org_df.to_csv(out_path, index=False)

    # Summary
    print(f"\n✓ Wrote {len(org_df)} rows to {out_path}")
    print(f"\nStatus value counts:")
    print(org_df["status"].value_counts().to_string())
    empty_desc = (org_df["description"] == "").sum()
    missing_count = (org_df["status"] == "missing").sum()
    print(f"\nRows with empty description: {empty_desc}")
    print(f"Rows with status 'missing':  {missing_count}")
    if empty_desc == missing_count:
        print("✓ Empty descriptions match missing-status count")
    else:
        print("⚠ MISMATCH — empty descriptions ≠ missing-status count")


if __name__ == "__main__":
    main()
