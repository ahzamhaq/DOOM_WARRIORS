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

## Setup (Python 3.12)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Unzip `student_resource.zip` into `data/raw/` so that `data/raw/dataset/{train,test}/` exists.

## Run (from repo root)
```bash
python -m notebooks.eda      # dataset report
python -m src.predict test --dry-run   # empty valid submission -> outputs/ (smoke test)
python -m src.predict train [--country C] [--limit-s1 N]   # dev world: train + tune threshold -> outputs/model.joblib
python -m src.predict dev   [--country C] [--limit-s1 N]   # dev world: validation macro F0.5 -> outputs/dev_report.md
python -m src.predict test             # full test run with the saved model -> outputs/*.tsv
python -m scripts.build_dev_world      # small frozen dev dataset -> data/processed/dev_world/ (~20 s)
python -m unittest discover -s tests -t . -v   # toy end-to-end contract test
python data/raw/utils/validate_submission.py -m outputs/matching_results.tsv -c outputs/candidate_pairs.tsv -t data/raw/dataset/test
```

## Pipeline & owners
| Module | Role | Owner |
| --- | --- | --- |
| `config.py`, `data_loader.py`, `predict.py` | paths, I/O, end-to-end integration | Lead |
| `normalize.py`, `blocking.py` | cleaning + candidate generation (`[s1_id, cand_id]`) | Person 2 |
| `features.py`, `matcher.py` | pair features + LightGBM + threshold | Person 3 |
| `evaluate.py` | macro F0.5, blocking recall, validation split, submission checks | Person 4 |

Read `docs/CONTRACT.md` before writing code: it fixes the data shapes between modules.

Constraints: only the supplied dataset is used, with no external APIs or data.
