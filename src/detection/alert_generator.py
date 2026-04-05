"""CRITICAL/HIGH alert filter — generates alerts for high-risk controls."""

import os
from datetime import datetime

import pandas as pd


def main():
    # Load risk register
    risk_path = os.path.join("data", "processed", "risk_register.csv")
    risk_df = pd.read_csv(risk_path)
    print(f"Loaded {len(risk_df)} rows from {risk_path}")

    # Filter CRITICAL and HIGH only
    alerts_df = risk_df[risk_df["risk_level"].isin(["CRITICAL", "HIGH"])].copy()

    # Sort by risk_score descending
    alerts_df = alerts_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    # Timestamp — same for all rows in one run
    created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Build output rows
    rows = []
    for i, (_, row) in enumerate(alerts_df.iterrows(), start=1):
        rows.append({
            "alert_id": f"ALT-{i:03d}",
            "control_id": row["control_id"],
            "control_family": row["control_family"],
            "risk_level": row["risk_level"],
            "risk_score": row["risk_score"],
            "anomaly_score": row["anomaly_score"],
            "status": row["status"],
            "owner": row["owner"],
            "alert_message": (
                f"{row['control_id']} ({row['control_family']}) is {row['status']}"
                f" \u2014 risk score {row['risk_score']}/10. Immediate review required."
            ),
            "created_at": created_at,
        })

    out_df = pd.DataFrame(rows)

    # Write
    out_path = os.path.join("data", "processed", "alerts.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # Summary
    critical_count = (out_df["risk_level"] == "CRITICAL").sum()
    high_count = (out_df["risk_level"] == "HIGH").sum()
    print(f"\n✓ Wrote {len(out_df)} alerts to {out_path}")
    print(f"  CRITICAL: {critical_count}")
    print(f"  HIGH:     {high_count}")
    print(f"\nTop 3 highest risk_score rows:")
    for _, r in out_df.head(3).iterrows():
        print(f"  {r['control_id']:12s}  risk_score={r['risk_score']}")


if __name__ == "__main__":
    main()
