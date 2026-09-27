# DOOM_WARRIORS — Progress

## Current Status

- [x] Repository/setup
- [x] Dataset setup
- [x] EDA
- [x] Integration contract
- [x] Dev world
- [x] Contract tests
- [ ] Normalization
- [ ] Blocking
- [ ] Feature engineering
- [ ] Matching model
- [ ] Validation
- [ ] Threshold tuning
- [ ] Test inference
- [ ] Submission files
- [ ] Official validator (passes today only on empty `--dry-run` output)
- [ ] Final ZIP
- [ ] Leaderboard submission

## Team Status

### Person 1 — Lead / Integration
- **Status:** In progress
- **Current task:** none (waiting on Persons 2/3 modules)
- **Completed:** repo + `.gitignore`; `config.py`, `data_loader.py`, `evaluate.py` (metric only); EDA (`notebooks/eda.py`, report in `outputs/eda_report.txt`, git-ignored); `docs/CONTRACT.md`; `predict.run_pipeline` with injectable components; frozen dev world (`scripts/build_dev_world.py`); toy contract test (`tests/test_contract.py`)
- **In progress:** —
- **Blockers:** none
- **Latest result:** `python -m src.predict test --dry-run` passes the official validator on the full 1,732,544-row test set; toy contract test (5/5) passes and is mutation-checked
- **Next action:** wire `predict.py` `train`/`dev` modes once `matcher.train` exists

### Person 2 — Normalization / Blocking
- **Status:** Not started
- **Current task:** `normalize.py` (`normalize_records`)
- **Completed:** —
- **In progress:** —
- **Blockers:** none — dev world and raw records are ready to use
- **Latest result:** —
- **Next action:** implement `normalize_records`, then `blocking.build_index` / `generate_candidates` per `docs/CONTRACT.md` §3–4

### Person 3 — Features / Matching
- **Status:** Not started
- **Current task:** `features.py` (`build_features`)
- **Completed:** —
- **In progress:** —
- **Blockers:** none — can build a fake `pairs` frame from dev-world ground truth (positives + random same-country negatives) until real blocking lands
- **Latest result:** —
- **Next action:** implement `FEATURE_NAMES` + `build_features`, then `matcher.train`; compare `ONE_TO_ONE` on/off once a model exists

### Person 4 — Evaluation / Submission
- **Status:** Not started
- **Current task:** `evaluate.py` remaining stubs
- **Completed:** (pre-existing) `f_beta`, `macro_f_beta`, `candidate_recall`, `truth_dict`
- **In progress:** —
- **Blockers:** none
- **Next action:** `split_s1`, `label_pairs`, `check_pairs`, `check_feats`, `blocking_report`, `sweep_thresholds`, `per_country_report` per `docs/CONTRACT.md` §7

## Metrics

| Experiment | Candidate Recall | Precision | Recall | F0.5 | Threshold | Notes |
| ---------- | ---------------: | --------: | -----: | ---: | --------: | ----- |

_No experiments run yet. Add a row only with measured numbers (dev world or larger); note which dataset was used._

## Integration Status

- [x] normalize → blocking *(contract only — verified with fakes in tests/test_contract.py, not real modules)*
- [x] blocking → features *(contract only — same)*
- [x] features → matcher *(contract only — same)*
- [ ] matcher → evaluate
- [x] matcher → predict *(contract only — fakes)*
- [x] predict → matching_results.tsv *(dry-run + toy test, real content pending)*
- [x] predict → candidate_pairs.tsv *(dry-run + toy test, real content pending)*
- [x] official validator passes *(on dry-run / toy output only, not a real prediction)*

## Decisions

- **2026-09-27 — `ONE_TO_ONE` defaults to `False`.** Reason: observed in training ground truth (each S2/S3 record matched ≤1 S1 entity) but not a stated competition rule; test/France may differ. Person 3 must compare validation F0.5 with/without before it is flipped. Responsible: Person 1.
- **2026-09-27 — Candidate generation blocks strictly by country.** Reason: EDA showed 100% of true pairs agree on country in the sampled analysis. Responsible: Person 1 (contract), Person 2 (implementation).
- **2026-09-27 — Dev-world size fixed at 30,000 S1 (seed 42), 20% val split.** Reason: fast (~20s build), covers both train countries, matches singleton rate (5.5% vs 5.6% full). Caveat: confusers thinned to ~1/73 density — recall/precision on it will read optimistic. Responsible: Person 1.
- **2026-09-25 — Repo made private.** Reason: dataset/competition rules; public repo risked exposing data or code during the challenge. Responsible: Person 1 (repo owner).

## Blockers

_None currently._

## Recent Changes

| Commit | Date | Change | Owner |
| --- | --- | --- | --- |
| 3d893d1 | 2026-09-27 | Integration contract, frozen dev-world builder, toy end-to-end contract test | Person 1 |
| f887429 | 2026-09-27 | Pin dependency versions (Python 3.12) | Person 1 |
| ce92f02 | 2026-09-27 | Add CLAUDE.md | Person 1 |
| 75d27bb | 2026-09-27 | Data loader, metric, EDA script, pipeline module stubs | Person 1 |
| fcd05dc | 2026-09-25 | Initial project scaffold | Person 1 |

## Next Milestones

1. Working blocking
2. Working matching
3. Validation/F0.5
4. End-to-end integration
5. Full test inference
6. Official validator
7. Final submission package
8. Leaderboard submission
