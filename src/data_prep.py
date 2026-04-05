"""NIST dataset cleaner — extracts control IDs and metadata from raw parquet files."""

import os
import re
import pandas as pd
from tqdm import tqdm

# ── FAMILY_MAP: family_code → (control_family, nist_csf_function, nist_csf_category) ──
FAMILY_MAP = {
    "AC": ("Access Control",         "PROTECT",  "PR.AC"),
    "AT": ("Awareness & Training",   "PROTECT",  "PR.AT"),
    "AU": ("Audit & Accountability", "DETECT",   "DE.CM"),
    "CA": ("Assessment & Auth",      "IDENTIFY", "ID.GV"),
    "CM": ("Config Management",      "PROTECT",  "PR.IP"),
    "CP": ("Contingency Planning",   "RECOVER",  "RC.RP"),
    "IA": ("Identification & Auth",  "PROTECT",  "PR.AC"),
    "IR": ("Incident Response",      "RESPOND",  "RS.RP"),
    "MA": ("Maintenance",            "PROTECT",  "PR.MA"),
    "MP": ("Media Protection",       "PROTECT",  "PR.DS"),
    "PE": ("Physical & Env",         "PROTECT",  "PR.IP"),
    "PL": ("Planning",               "IDENTIFY", "ID.GV"),
    "PM": ("Program Management",     "IDENTIFY", "ID.RM"),
    "PS": ("Personnel Security",     "PROTECT",  "PR.AT"),
    "RA": ("Risk Assessment",        "IDENTIFY", "ID.RA"),
    "SA": ("System Acquisition",     "PROTECT",  "PR.IP"),
    "SC": ("System & Comms",         "PROTECT",  "PR.PT"),
    "SI": ("System Integrity",       "DETECT",   "DE.CM"),
    "SR": ("Supply Chain Risk",      "IDENTIFY", "ID.SC"),
}


def load_nist_data(split: str) -> pd.DataFrame:
    """Read data/raw/nist_{split}.parquet. Raises FileNotFoundError with download
    instructions if the file is missing."""
    path = os.path.join("data", "raw", f"nist_{split}.parquet")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Download the dataset first:\n"
            "  python -c \"\n"
            "  from datasets import load_dataset\n"
            "  ds = load_dataset('ethanolivertroy/nist-cybersecurity-training')\n"
            "  ds['train'].to_parquet('data/raw/nist_train.parquet')\n"
            "  ds['validation'].to_parquet('data/raw/nist_val.parquet')\n"
            "  \""
        )
    return pd.read_parquet(path)


def extract_controls(text: str, row_id: int) -> list[dict]:
    """Parse a single NIST text row into a list of control dicts.

    The text column format is:
        "You are a cybersecurity expert... [QUESTION] ; [ANSWER]"
    We split on " ; " (space-semicolon-space) and use the answer portion
    as the description. Control IDs are extracted via regex.
    """
    # Split on " ; " to separate question from answer
    parts = text.split(" ; ", maxsplit=1)
    description = parts[1].strip() if len(parts) > 1 else text.strip()

    # Trim description to 500 characters
    description = description[:500]

    # Extract control IDs
    control_ids = re.findall(r'\b([A-Z]{2}-\d+(?:\(\d+\))?)\b', text)

    results = []
    seen = set()
    for cid in control_ids:
        if cid in seen:
            continue
        seen.add(cid)

        family_code = cid.split("-")[0]
        if family_code not in FAMILY_MAP:
            continue

        control_family, nist_csf_function, nist_csf_category = FAMILY_MAP[family_code]
        results.append({
            "control_id": cid,
            "family_code": family_code,
            "control_family": control_family,
            "nist_csf_function": nist_csf_function,
            "nist_csf_category": nist_csf_category,
            "description": description,
            "source_row_id": row_id,
        })

    return results


def main():
    """Process both train and validation parquets, deduplicate, write controls_clean.csv."""
    all_records = []

    for split in ("train", "val"):
        print(f"\n── Loading {split} split ──")
        df = load_nist_data(split)
        print(f"  Rows in {split}: {len(df)}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            controls = extract_controls(text, row_id=idx)
            all_records.extend(controls)

    print(f"\nTotal raw control records extracted: {len(all_records)}")

    # Build DataFrame
    controls_df = pd.DataFrame(all_records)

    if controls_df.empty:
        print("ERROR: No controls extracted. Check the parquet data.")
        return

    # Deduplicate on control_id — keep the row with the longest description
    controls_df["desc_len"] = controls_df["description"].str.len()
    controls_df = controls_df.sort_values("desc_len", ascending=False)
    controls_df = controls_df.drop_duplicates(subset=["control_id"], keep="first")
    controls_df = controls_df.drop(columns=["desc_len"])

    # Sort by control_id for deterministic output
    controls_df = controls_df.sort_values("control_id").reset_index(drop=True)

    # Ensure column order
    col_order = [
        "control_id", "family_code", "control_family",
        "nist_csf_function", "nist_csf_category",
        "description", "source_row_id",
    ]
    controls_df = controls_df[col_order]

    # Write output
    out_path = os.path.join("data", "processed", "controls_clean.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    controls_df.to_csv(out_path, index=False)

    print(f"\n✓ Wrote {len(controls_df)} unique controls to {out_path}")
    print(f"  Unique control_ids: {controls_df['control_id'].nunique()}")
    print(f"\nFamily code value counts:")
    print(controls_df["family_code"].value_counts().sort_index().to_string())
    print(f"\nNull descriptions: {controls_df['description'].isna().sum()}")


if __name__ == "__main__":
    main()
