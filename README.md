# CyberGuard

CyberGuard is a 100% local GRC system. It uses an AI-driven anomaly model to dynamically score risks based on configuration data aligned with NIST controls, running entirely locally without external API calls at runtime.

## Quick Start

```bash
pip install -r requirements.txt
python -c "from datasets import load_dataset; ds = load_dataset('ethanolivertroy/nist-cybersecurity-training'); ds['train'].to_parquet('data/raw/nist_train.parquet'); ds['validation'].to_parquet('data/raw/nist_val.parquet')"
python src/pipeline.py
streamlit run src/dashboard/app.py
```
