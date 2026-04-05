"""NIST↔ISO↔SOC2 crosswalk builder.

Produces data/processed/crosswalk.csv with one row per control_id from
controls_clean.csv. Uses a hardcoded seed table for DIRECT mappings and
TF-IDF cosine similarity for INFERRED mappings.
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── FAMILY_MAP (mirrors data_prep.py) ──────────────────────────────────────────
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

# ── HARDCODED SEED TABLE (all DIRECT) ─────────────────────────────────────────
SEED_TABLE = [
    {"nist_800_53_id": "AC-2",  "nist_csf_id": "PR.AC-1", "nist_csf_function": "PROTECT",  "iso_27001_id": "5.16", "iso_27001_name": "Identity management",           "soc2_criteria": "CC6.1", "soc2_name": "Logical access security"},
    {"nist_800_53_id": "AC-3",  "nist_csf_id": "PR.AC-4", "nist_csf_function": "PROTECT",  "iso_27001_id": "5.15", "iso_27001_name": "Access control",                "soc2_criteria": "CC6.1", "soc2_name": "Logical access security"},
    {"nist_800_53_id": "AC-17", "nist_csf_id": "PR.AC-3", "nist_csf_function": "PROTECT",  "iso_27001_id": "8.20", "iso_27001_name": "Networks security",             "soc2_criteria": "CC6.6", "soc2_name": "Network and transmission controls"},
    {"nist_800_53_id": "AU-2",  "nist_csf_id": "DE.CM-1", "nist_csf_function": "DETECT",   "iso_27001_id": "8.15", "iso_27001_name": "Logging",                       "soc2_criteria": "CC7.2", "soc2_name": "System monitoring"},
    {"nist_800_53_id": "IA-2",  "nist_csf_id": "PR.AC-7", "nist_csf_function": "PROTECT",  "iso_27001_id": "5.17", "iso_27001_name": "Authentication information",    "soc2_criteria": "CC6.1", "soc2_name": "Logical access security"},
    {"nist_800_53_id": "IA-5",  "nist_csf_id": "PR.AC-1", "nist_csf_function": "PROTECT",  "iso_27001_id": "5.17", "iso_27001_name": "Authentication information",    "soc2_criteria": "CC6.1", "soc2_name": "Logical access security"},
    {"nist_800_53_id": "IR-4",  "nist_csf_id": "RS.RP-1", "nist_csf_function": "RESPOND",  "iso_27001_id": "5.26", "iso_27001_name": "Response to IS incidents",      "soc2_criteria": "CC7.3", "soc2_name": "Incident response"},
    {"nist_800_53_id": "RA-3",  "nist_csf_id": "ID.RA-1", "nist_csf_function": "IDENTIFY", "iso_27001_id": "8.2",  "iso_27001_name": "Information security risk",     "soc2_criteria": "CC3.1", "soc2_name": "Risk assessment"},
    {"nist_800_53_id": "RA-5",  "nist_csf_id": "ID.RA-1", "nist_csf_function": "IDENTIFY", "iso_27001_id": "8.8",  "iso_27001_name": "Management of vulnerabilities", "soc2_criteria": "CC7.1", "soc2_name": "System vulnerability"},
    {"nist_800_53_id": "SC-7",  "nist_csf_id": "PR.PT-4", "nist_csf_function": "PROTECT",  "iso_27001_id": "8.22", "iso_27001_name": "Web filtering",                 "soc2_criteria": "CC6.6", "soc2_name": "Network and transmission controls"},
    {"nist_800_53_id": "SC-28", "nist_csf_id": "PR.DS-1", "nist_csf_function": "PROTECT",  "iso_27001_id": "8.24", "iso_27001_name": "Use of cryptography",           "soc2_criteria": "CC6.1", "soc2_name": "Logical access security"},
    {"nist_800_53_id": "SI-2",  "nist_csf_id": "ID.RA-1", "nist_csf_function": "IDENTIFY", "iso_27001_id": "8.8",  "iso_27001_name": "Management of vulnerabilities", "soc2_criteria": "CC7.1", "soc2_name": "System vulnerability"},
    {"nist_800_53_id": "SI-3",  "nist_csf_id": "DE.CM-4", "nist_csf_function": "DETECT",   "iso_27001_id": "8.7",  "iso_27001_name": "Protection against malware",    "soc2_criteria": "CC6.8", "soc2_name": "Anti-malware"},
    {"nist_800_53_id": "CP-9",  "nist_csf_id": "RC.RP-1", "nist_csf_function": "RECOVER",  "iso_27001_id": "8.13", "iso_27001_name": "Information backup",            "soc2_criteria": "A1.2",  "soc2_name": "Recovery planning"},
]


def main():
    # ── 1. Load controls_clean.csv ─────────────────────────────────────────────
    controls_path = os.path.join("data", "processed", "controls_clean.csv")
    controls_df = pd.read_csv(controls_path)
    print(f"Loaded {len(controls_df)} controls from {controls_path}")

    # Build lookup: control_id → row
    ctrl_lookup = controls_df.set_index("control_id").to_dict("index")

    # ── 2. Index seed table by nist_800_53_id ──────────────────────────────────
    seed_map = {row["nist_800_53_id"]: row for row in SEED_TABLE}
    seeded_ids = set(seed_map.keys())

    # ── 3. Build TF-IDF matrix on ALL control descriptions ─────────────────────
    all_ids = controls_df["control_id"].tolist()
    all_descs = controls_df["description"].fillna("").tolist()

    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_descs)

    # Get indices for seeded controls (that exist in controls_clean)
    seed_indices = []
    seed_ids_present = []
    for i, cid in enumerate(all_ids):
        if cid in seeded_ids:
            seed_indices.append(i)
            seed_ids_present.append(cid)

    print(f"Seed controls found in controls_clean: {len(seed_indices)} / {len(SEED_TABLE)}")

    # Precompute seed TF-IDF sub-matrix
    if seed_indices:
        seed_tfidf = tfidf_matrix[seed_indices]

    # ── 4. Build crosswalk rows ────────────────────────────────────────────────
    crosswalk_rows = []

    for i, cid in enumerate(all_ids):
        if cid in seed_map:
            # DIRECT mapping
            row = dict(seed_map[cid])
            row["mapping_strength"] = "DIRECT"
            crosswalk_rows.append(row)
        else:
            # INFERRED mapping
            ctrl_info = ctrl_lookup.get(cid, {})
            family_code = ctrl_info.get("family_code", cid.split("-")[0])

            # Derive nist_csf_id and nist_csf_function from FAMILY_MAP
            if family_code in FAMILY_MAP:
                _, nist_csf_function, nist_csf_category = FAMILY_MAP[family_code]
                nist_csf_id = nist_csf_category
            else:
                nist_csf_function = "IDENTIFY"
                nist_csf_id = "TBD"

            # Compute cosine similarity against seed controls
            if seed_indices:
                sim_scores = cosine_similarity(
                    tfidf_matrix[i:i+1], seed_tfidf
                ).flatten()
                max_sim = sim_scores.max()
                best_seed_idx = sim_scores.argmax()
                best_seed_id = seed_ids_present[best_seed_idx]
            else:
                max_sim = 0.0
                best_seed_id = None

            if max_sim >= 0.15 and best_seed_id is not None:
                # Copy ISO/SOC2 from closest seed
                best_seed = seed_map[best_seed_id]
                iso_id = best_seed["iso_27001_id"]
                iso_name = best_seed["iso_27001_name"]
                soc2_crit = best_seed["soc2_criteria"]
                soc2_name = best_seed["soc2_name"]
            else:
                iso_id = "TBD"
                iso_name = "TBD"
                soc2_crit = "TBD"
                soc2_name = "TBD"

            crosswalk_rows.append({
                "nist_800_53_id": cid,
                "nist_csf_id": nist_csf_id,
                "nist_csf_function": nist_csf_function,
                "iso_27001_id": iso_id,
                "iso_27001_name": iso_name,
                "soc2_criteria": soc2_crit,
                "soc2_name": soc2_name,
                "mapping_strength": "INFERRED",
            })

    # ── 5. Write crosswalk.csv ─────────────────────────────────────────────────
    col_order = [
        "nist_800_53_id", "nist_csf_id", "nist_csf_function",
        "iso_27001_id", "iso_27001_name",
        "soc2_criteria", "soc2_name", "mapping_strength",
    ]
    cross_df = pd.DataFrame(crosswalk_rows)[col_order]

    out_path = os.path.join("data", "processed", "crosswalk.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cross_df.to_csv(out_path, index=False)

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    print(f"\n✓ Wrote {len(cross_df)} rows to {out_path}")
    print(f"  DIRECT   mappings: {(cross_df['mapping_strength'] == 'DIRECT').sum()}")
    print(f"  INFERRED mappings: {(cross_df['mapping_strength'] == 'INFERRED').sum()}")
    print(f"  TBD iso_27001_id:  {(cross_df['iso_27001_id'] == 'TBD').sum()}")

    # Sanity check — every control_id from controls_clean must be present
    missing = set(all_ids) - set(cross_df["nist_800_53_id"])
    if missing:
        print(f"\n⚠ WARNING: {len(missing)} control_ids missing from crosswalk: {missing}")
    else:
        print("\n✓ All control_ids from controls_clean.csv are present in crosswalk.csv")


if __name__ == "__main__":
    main()
