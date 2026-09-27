# DOOM_WARRIORS — Business Entity Resolution (Amazon ML Challenge 2026)

Self-contained, runnable copy of our pipeline. It regenerates `output/matching_results.tsv` and
`output/candidate_pairs.tsv` from the competition's training and test data only. No external data, APIs or
pretrained models are used (LightGBM, MIT licence, trained from scratch on the supplied training data).

## 1. Environment
Python 3.12. From this folder (`code/business_entity_resolution/`):
```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # pinned versions
```

## 2. Data
Unzip the competition's `student_resource.zip` so that these files exist:
```
data/raw/dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
data/raw/dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```
(i.e. copy the zip's `dataset/` folder into `data/raw/` here). All paths are defined in `src/config.py`.

## 3a. Reproduce the SUBMITTED outputs (fast exact-key matcher, ~7 min)
```bash
python -m src.quick_match test    # -> outputs/matching_results.tsv, outputs/candidate_pairs.tsv
python -m src.quick_match dev     # optional: macro F0.5 on the dev world (0.54, pair precision 0.9985)
```
Rule (`src/quick_match.py`): within the same country, the cleaned name (lowercased, punctuation and legal-form
words removed) and the first house number of the address must be identical; keys shared by more than 10 pool
records are dropped. `candidate_pairs.tsv` equals the matched pairs for this rule.

## 3b. Full ML pipeline (normalization → blocking → features → LightGBM → threshold)
```bash
python -m src.build_dev_world     # deterministic training subset (seed 42) -> data/processed/dev_world/
python -m src.predict train       # train LightGBM, tune the F0.5 threshold on held-out S1 -> outputs/model.joblib
python -m src.predict dev         # optional: validation macro F0.5 -> outputs/dev_report.md
python -m src.predict test        # full test set -> outputs/matching_results.tsv, outputs/candidate_pairs.tsv
```
To reproduce the submitted outputs **without retraining**, skip `train`: the exact model used for the submission
is included as `outputs/model.joblib` (model + tuned threshold), and `python -m src.predict test` uses it.

`outputs/candidate_pairs.tsv` is written by the same run: it is exactly the candidate set the matcher scores.
The test set is processed one country at a time (countries read from the data) in chunks of S1 entities.

## 4. Pipeline (where to look)
| Step | File |
| --- | --- |
| Loading TSVs | `src/data_loader.py` |
| Name / address normalization | `src/normalize.py` |
| Candidate generation (blocking) | `src/blocking.py` |
| Pairwise features | `src/features.py` |
| Classifier, threshold, final match selection | `src/matcher.py` |
| Metric (macro F0.5), validation helpers | `src/evaluate.py` |
| End-to-end driver (train / dev / test) | `src/predict.py` |
| Deterministic training subset | `src/build_dev_world.py` |
| Settings (paths, chunk size, K, grid, seed) | `src/config.py` |

Methodology: see `Documentation_template.md` at the root of the submission zip.
