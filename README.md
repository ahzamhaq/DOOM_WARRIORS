# DOOM_WARRIORS — Amazon ML Challenge 2026

Business entity resolution across three noisy data sources. Metric: macro F0.5.

## Layout
```
data/raw/        # competition files (git-ignored, place unzipped dataset here)
data/processed/  # intermediate features (git-ignored)
src/             # pipeline code
notebooks/       # EDA
outputs/         # submissions & models (git-ignored)
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Constraints: only the supplied dataset is used, with no external APIs or data.
