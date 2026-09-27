# business_entity_resolution: reproduction steps

This code produces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the competition dataset
only. It makes no network calls and uses no external data.

## 1. Environment (Python 3.12)

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # exact pinned versions
```

## 2. Data

Put the competition files in `data/raw/`, so that these paths exist:

```
data/raw/dataset/train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
data/raw/dataset/test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
data/raw/utils/validate_submission.py
```

## 3. Train, validate, predict (run from this folder)

```bash
python -m scripts.build_dev_world   # deterministic training subset -> data/processed/dev_world/ (~20 s)
python -m src.predict train         # candidates -> features -> LightGBM -> threshold -> outputs/model.joblib
python -m src.predict dev           # validation macro F0.5 + blocking / per-country reports
python -m src.predict test          # full test set -> outputs/matching_results.tsv, outputs/candidate_pairs.tsv
```

[[TBD: confirm the final training command and data size (dev world vs larger run) once predict train is merged.]]

The pipeline runs one country at a time, in chunks of 50,000 Source 1 records, and fits in about 8 GB of RAM.
Seed 42 is used everywhere.

## 4. Validate the output

```bash
python data/raw/utils/validate_submission.py -m outputs/matching_results.tsv -c outputs/candidate_pairs.tsv -t data/raw/dataset/test --check-ids
```

`python -m scripts.package_submission` runs these checks, plus its own, and then builds the submission zip.
If anything fails, it builds nothing.

## Layout

| Path | Contents |
| --- | --- |
| `src/config.py` | paths, column names, tuning constants, seed |
| `src/data_loader.py` | TSV loading (tab-separated, empty string for missing values) and output writers |
| `src/normalize.py` | name/address normalization |
| `src/blocking.py` | per-country candidate generation (name + address character n-gram retrieval) |
| `src/features.py` | 21 pairwise similarity features |
| `src/matcher.py` | LightGBM classifier, threshold, match selection |
| `src/evaluate.py` | macro F0.5, validation split, candidate recall, contract checks |
| `src/predict.py` | end-to-end driver (train / dev / test) |
| `scripts/` | dev-world builder, submission packager |
| `notebooks/` | EDA, matcher training report, evaluation cross-checks |
