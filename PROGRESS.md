# DOOM_WARRIORS — Progress

## Current Status

- [x] Repository/setup
- [x] Dataset setup
- [x] EDA
- [x] Integration contract
- [x] Dev world
- [x] Contract tests
- [ ] Normalization *(implemented on `feature/blocking`, commit 94a622c; no PR yet, not reviewed)*
- [ ] Blocking *(`blocking.py` still a stub)*
- [x] Feature engineering *(merged in PR #1)*
- [ ] Matching model *(LightGBM matcher merged in PR #1; final model not trained yet: waits for real normalization + blocking)*
- [ ] Validation *(Person 3's own validation run exists; `evaluate.py` validation helpers still stubs)*
- [ ] Threshold tuning *(provisional 0.70 on stand-in blocking; re-tune after the real pipeline)*
- [ ] Test inference
- [ ] Submission files
- [ ] Official validator *(passes on `--dry-run` and on toy output with the real matcher; no real prediction yet)*
- [ ] Final ZIP
- [ ] Leaderboard submission

## Team Status

### Person 1 — Lead / Integration
- **Status:** In progress
- **Current task:** waiting for Person 2's normalization/blocking PR
- **Completed:** repo, `config.py`, `data_loader.py`, metric in `evaluate.py`, EDA, `docs/CONTRACT.md`, `predict.run_pipeline`, frozen dev world, toy contract test; reviewed, verified and merged PR #1; PR #2 (TF-IDF context once per country) merged after Person 3's approval
- **In progress:** —
- **Blockers:** none
- **Latest result:** PR #2 merged (6a62993); on merged `main`: 15/15 tests pass, `--dry-run` passes the official validator. PR #2 measurements: identical features with/without `ctx` on India dev (454,666 pairs); 139.9 s → 56.9 s over 7 chunks (13.8 s saved per extra chunk)
- **Next action:** review Person 2's normalization/blocking once a PR is opened; add `train`/`dev` modes to `predict.py`; optional pool-fingerprint check in `ctx` (deferred follow-up)

### Person 2 — Normalization / Blocking
- **Status:** In progress
- **Current task:** normalization pushed; blocking next
- **Completed:** —  *(nothing merged yet)*
- **In progress:** `normalize.py` + `tests/test_normalize.py` on `feature/blocking` (commit 94a622c, no PR, not reviewed); `blocking.py` not started on that branch
- **Blockers:** none
- **Latest result:** —
- **Next action:** open a PR for normalization (or together with blocking); implement `build_index` / `generate_candidates` per `docs/CONTRACT.md` §4 and report candidate recall on the dev world

### Person 3 — Features / Matching
- **Status:** In progress
- **Current task:** waiting for real blocking to retrain
- **Completed:** `features.py` (21 features, `anyascii` transliteration, per-unique-record vectorization), `matcher.py` (LightGBM, monotone constraints, save/load, `select_matches` with optional one-to-one), `notebooks/train_matcher.py`, `notebooks/bench_features.py`, 8 unit tests: merged in PR #1; reviewed and approved PR #2
- **In progress:** —
- **Blockers:** final training/threshold depends on Person 2's real normalization + blocking
- **Latest result:** dev world + 60 hard decoys, stand-in blocking: macro F0.5 0.9409 (reported); reproduced by Lead as 0.9400. Features: 59 µs/pair and ~0.5 GB per 50k-S1 chunk measured on the Lead's laptop
- **Next action:** retrain + re-tune once real blocking is merged; check false merges from shared addresses (address features ≈55% of gain)

### Person 4 — Evaluation / Submission
- **Status:** No commits pushed yet (`feature/evaluation` has none)
- **Current task:** `evaluate.py` remaining stubs
- **Completed:** (pre-existing) `f_beta`, `macro_f_beta`, `candidate_recall`, `truth_dict`
- **In progress:** —
- **Blockers:** none
- **Latest result:** —
- **Next action:** `split_s1`, `label_pairs`, `check_pairs`, `check_feats`, `blocking_report`, `sweep_thresholds`, `per_country_report` per `docs/CONTRACT.md` §7 (`blocking_report` is needed as soon as Person 2's blocking lands)

## Metrics

All rows: frozen dev world (seed 42) + hard decoys (60 per name word), **stand-in blocking** (not Person 2's), 6,044 validation S1, one-to-one OFF. Numbers will change with the real pipeline.

| Experiment | Candidate Recall | Precision | Recall | F0.5 | Threshold | Notes |
| ---------- | ---------------: | --------: | -----: | ---: | --------: | ----- |
| PR #1 matcher, reported by Person 3 | — | 0.9758 | 0.8914 | 0.9409 | 0.70 | US 0.958, India 0.916; LOCO US→India 0.889, India→US 0.941 |
| PR #1 matcher, reproduced by Lead (8a79a39) | 0.9509 | 0.9757 | 0.8902 | 0.9400 | 0.70 | US 0.957, India 0.915; LOCO US→India 0.889, India→US 0.942; one-to-one ON: ±0.000 |

## Integration Status

- [ ] normalize → blocking *(contract + fakes only; neither real module is merged)*
- [ ] blocking → features *(contract + fakes only; features side is real)*
- [x] features → matcher *(real modules, verified in PR #1 review)*
- [ ] matcher → evaluate *(`evaluate.py` helpers not implemented)*
- [x] matcher → predict *(real matcher through `run_pipeline`, verified in PR #1 review)*
- [x] predict → matching_results.tsv *(real matcher on toy data + `--dry-run`; no real test prediction yet)*
- [x] predict → candidate_pairs.tsv *(same)*
- [x] official validator passes *(on `--dry-run` and toy output with the real matcher only)*

## Decisions

- **2026-09-27 — Pool-fingerprint check in `ctx` deferred.** Reason: suggested by Person 3 as optional; wrong-country `ctx` is not possible via `predict.py` (built and dropped per country, enforced by a contract test) and would affect only the two TF-IDF features. Revisit if `ctx` is used outside `predict.py`. Responsible: Person 1.
- **2026-09-27 — Final model is not retrained until Person 2's normalization + blocking are merged.** Reason: the current model was trained on stand-in blocking, and `blk_*` / normalized columns change meaning with the real modules. Responsible: Person 1.
- **2026-09-27 — `ONE_TO_ONE` stays `False`.** Reason: measured on validation, it changes macro F0.5 by +0.0007 (Person 3) / ±0.000 (Lead reproduction); not a stated competition rule. Responsible: Person 1.
- **2026-09-27 — TF-IDF corpus statistics are computed once per country (`features.build_context`).** Reason: refitting per chunk cost 13.8 s per chunk on this laptop. Contract updated first; PR #2 approved by Person 3 (bit-identical features, ctx 8 MB) and merged. Responsible: Person 1 (contract), Person 3 (review).
- **2026-09-27 — `anyascii==0.3.3` is a required dependency.** Reason: without it 16.9% of true India pairs lost all name features (Indic-script names became empty). ISC licence; generic Unicode transliteration tables, not external business data. Every teammate must re-run `pip install -r requirements.txt`. Responsible: Person 1 (approved in PR #1 review).
- **2026-09-27 — PR #1 (features + matcher) merged.** Reason: passed contract, integration, official-validator, performance and memory checks after review fixes. Responsible: Person 1.
- **2026-09-27 — `ONE_TO_ONE` defaults to `False`.** Reason: observed in training ground truth (each S2/S3 record matched ≤1 S1 entity) but not a stated competition rule; test/France may differ. Responsible: Person 1.
- **2026-09-27 — Candidate generation blocks strictly by country.** Reason: EDA showed 100% of true pairs agree on country in the sampled analysis. Responsible: Person 1 (contract), Person 2 (implementation).
- **2026-09-27 — Dev-world size fixed at 30,000 S1 (seed 42), 20% val split.** Reason: fast (~20 s build), both train countries, singleton rate 5.5% vs 5.6% full. Caveat: confusers thinned to ~1/73 density, so metrics read optimistic. Responsible: Person 1.
- **2026-09-27 — Repo made private.** Reason: competition rules; a public repo risked exposing data or code during the challenge. Responsible: Person 1 (repo owner).

## Blockers

| Person | Problem | Impact | Required action |
| --- | --- | --- | --- |
| Person 3 | Final matcher training needs real candidates | Cannot retrain or set the final threshold; current 0.940 is on stand-in blocking | Person 2 opens a PR for normalization + blocking; Lead reviews and merges |

## Recent Changes

| Commit | Date | Change | Owner |
| --- | --- | --- | --- |
| 6a62993 | 2026-09-27 | Merge PR #2: TF-IDF context once per country (approved by Person 3) | Person 1 |
| fd83858 | 2026-09-27 | CLAUDE.md, .gitignore: repo is private | Person 1 |
| 31cb196 | 2026-09-27 | Update PROGRESS.md | Person 1 |
| 396062c | 2026-09-27 | TF-IDF context once per country (code + tests) | Person 1 |
| 8fcb62e | 2026-09-27 | CONTRACT: per-country feature context | Person 1 |
| 94a622c | 2026-09-27 | Normalization implementation (`feature/blocking`, no PR yet) | Person 2 |
| c4d464b | 2026-09-27 | Merge PR #1: features + LightGBM matcher | Person 1 |
| 8a79a39 | 2026-09-27 | Review fixes: `anyascii` required, per-record vectorization, KeyError on missing ids, unit tests | Person 3 |
| 5eb91b7 | 2026-09-27 | Pairwise features + LightGBM matcher | Person 3 |
| 9c7269c | 2026-09-27 | Add PROGRESS.md | Person 1 |

## Next Milestones

1. Working blocking
2. Working matching
3. Validation/F0.5
4. End-to-end integration
5. Full test inference
6. Official validator
7. Final submission package
8. Leaderboard submission
