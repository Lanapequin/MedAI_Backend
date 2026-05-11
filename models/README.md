# models/

This directory holds the trained MedAI triage model.

The `.joblib` artifact is **not committed to git** (binary, ~12 MB, regenerated on each training run).

## How to generate `triage_model.joblib`

```bash
cd MedAI_Backend

# Activate venv (first run: see main README for setup)
source venv/bin/activate   # Linux/macOS
# or
venv\Scripts\activate      # Windows

# Train — takes ~5 minutes on CPU
python train_model.py
```

The script reads `triage_choco_100k_balanceado.csv`, runs SMOTE resampling,
trains XGBoost with class-weight balancing, and writes:

```
models/triage_model.joblib    ← loaded by api/main.py at startup
```

## Dataset

`triage_choco_100k_balanceado.csv` contains 100,000 anonymised triage records
from the department of Chocó, Colombia, balanced across 5 MTS urgency levels.
It is committed to the repository for reproducibility.
