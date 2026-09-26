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

Unzip `student_resource.zip` into `data/raw/` so that `data/raw/dataset/{train,test}/` exists.

## Run (from repo root)
```bash
python -m notebooks.eda      # dataset report
python -m src.predict        # writes outputs/matching_results.tsv + candidate_pairs.tsv
python data/raw/utils/validate_submission.py -m outputs/matching_results.tsv -c outputs/candidate_pairs.tsv -t data/raw/dataset/test
```

## Pipeline & owners
| Module | Role | Owner |
| --- | --- | --- |
| `config.py`, `data_loader.py`, `predict.py` | paths, I/O, end-to-end integration | Lead |
| `normalize.py`, `blocking.py` | cleaning + candidate generation (`[s1_id, cand_id]`) | Person 2 |
| `features.py`, `matcher.py` | pair features + LightGBM + threshold | Person 3 |
| `evaluate.py` | macro F0.5, blocking recall, validation split, submission checks | Person 4 |

Constraints: only the supplied dataset is used, with no external APIs or data.
