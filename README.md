# CyberGuard

CyberGuard is a 100% local GRC system. It uses an AI-driven anomaly model to dynamically score risks based on configuration data aligned with NIST controls, running entirely locally without external API calls at runtime.

## Quick Start

```bash
pip install -r requirements.txt
python -c "from datasets import load_dataset; ds = load_dataset('ethanolivertroy/nist-cybersecurity-training'); ds['train'].to_parquet('data/raw/nist_train.parquet'); ds['validation'].to_parquet('data/raw/nist_val.parquet')"
python src/pipeline.py
streamlit run src/dashboard/app.py
```
## What it does

1. **Downloads** the NIST cybersecurity training dataset from HuggingFace (one time only)
2. **Trains** a local TF-IDF + IsolationForest model on 425k NIST control descriptions
3. **Maps** controls across frameworks: NIST 800-53 ↔ NIST CSF ↔ ISO 27001 ↔ SOC2
4. **Detects** anomalous and missing controls in your organisation's inventory
5. **Shows** a Streamlit dashboard with risk heatmap, gap analysis, and alerts

No cloud. No API calls at runtime. Entire model runs in `src/model/artifacts/anomaly_model.pkl`.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (one time, ~500MB)
python -c "
from datasets import load_dataset
ds = load_dataset('ethanolivertroy/nist-cybersecurity-training')
ds['train'].to_parquet('data/raw/nist_train.parquet')
ds['validation'].to_parquet('data/raw/nist_val.parquet')
print('Done.')
"

# 3. Run full pipeline (train model, build crosswalk, score controls)
python src/pipeline.py

# 4. Launch dashboard
streamlit run src/dashboard/app.py
```

Open browser at `http://localhost:8501`

---

## Dashboard Tabs

| Tab | Shows |
|-----|-------|
| Risk Heatmap | Control family vs severity matrix |
| Control Mapping | NIST ↔ ISO ↔ SOC2 crosswalk table |
| Gap Analysis | Stacked bar of missing/partial/planned by family |
| Anomaly Feed | ML-detected anomalous controls sorted by risk score |

---
