# CLAUDE.md — DOOM_WARRIORS, Amazon ML Challenge 2026

Persistent instructions for all work in this repository. Follow them unless the user explicitly says otherwise.

## Project context
Business **entity resolution** across three independent, noisy data sources.

- **Source 1 (S1)** is the deduplicated reference source.
- For every S1 entity, find the matching records in **Source 2 and Source 3**.
- An S1 entity may have zero, one, or many matches.
- Training data has ground truth; test data does not.
- Metric: **macro F0.5**, computed per S1 entity then averaged over all S1 entities. It is precision-heavy, so **false merges cost more than missed matches**. Singletons count: predicting an empty list for a true singleton scores 1.0, and any match predicted for it scores 0.0.

## Dataset
All files are **TSV**. Always read with `sep="\t"`; never assume CSV. Reading without it silently yields one column.

```
data/raw/dataset/
├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
└── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
data/raw/README.md               official problem statement (source of truth for rules)
data/raw/utils/validate_submission.py   official validator
```
(The unzip nests files under `dataset/`. Use `src/config.py` paths, not hard-coded strings.)

- Source columns: `entity_id`, `business_name`, `business_address`, `country`. The source is given by the ID prefix (`S1-`/`S2-`/`S3-`).
- Ground truth columns: `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 IDs, empty for singletons).
- Use `src/data_loader.py` for loading. It returns empty strings, never NaN, for missing fields.

### EDA findings (full report: `python -m notebooks.eda`, output in `outputs/eda_report.txt`)
- Rows: train S1 2.21M, S2 5.03M, S3 5.29M; test S1 1.73M, S2 4.89M, S3 5.08M.
- **Test contains France, which is absent from training.** Treat `country` as an open set of labels. **Never hard-code US/India** (no filters, no one-hot on country). Every test S1 entity, France included, must be processed.
- Matches per S1 entity: 5.6% have 0, 5.4% have 1, 89% have 2+ (mean 3.46, max 11). S2 and S3 contribute about equally.
- Every true match agreed on country. Each S2/S3 record matches at most one S1 entity. About 26% of S2/S3 records match nothing (decoys).
- Name-token blocking alone recovered only ~85% of sampled true pairs. Address search recovers important misses (other-script names, website-style names).
- Noise seen: junk prefixes (`--`, `<<`, `#`), legal-suffix variants, typos and lookalike characters, website/handle names, missing S2/S3 addresses (~3–4%), casing and format differences (S2 addresses are uppercase), Devanagari and other Indic scripts.
- **Postcodes are unreliable** (present in only ~11% of US and ~1% of India addresses). Never use them as the primary blocking key.

## Architecture
Load → normalize → candidate blocking → pairwise features → matching model → threshold selection on validation → test prediction → output validation → submission.

Candidate generation is country-aware:
1. Block by country.
2. Character n-gram similarity search on names.
3. Character n-gram similarity search on addresses.
4. Union the candidate sets.
5. Measure candidate recall on held-out training data.

Candidate features to try (test each, keep only what helps; none are mandatory): name/address similarity, character and token similarity, TF-IDF cosine, edit similarity, house-number agreement, missing-field flags, country agreement, candidate rank/score. Keep features country-agnostic so they transfer to France.

`candidate_pairs.tsv` must be the exact set fed to the matcher (the last stage before scoring), and matches must be a subset of it.

## Team and file ownership
| Person | Role | Owns |
| --- | --- | --- |
| 1 | Lead / integration | `predict.py`, `config.py`, `data_loader.py`; architecture, merging branches, end-to-end runs, final predictions, final validation |
| 2 | Normalization / blocking | `normalize.py`, `blocking.py`; high-recall candidates, measure candidate recall. Does not touch the matching model |
| 3 | Features / matching | `features.py`, `matcher.py`; pair features, model training/tuning, macro F0.5. Does not touch blocking |
| 4 | Evaluation / submission | `evaluate.py`; validation split, F0.5, threshold tests, output checks, methodology docs. No core matching/blocking changes without coordination |

Rules: respect file ownership; do not rewrite a teammate's module without explicit instruction; explain the reason before any broad architectural change. Module interfaces are documented in each stub's docstring; do not change them silently.

## Memory constraints (important)
The dataset is huge (~11.7M test records) and the dev machine has little free RAM (~7.7 GB total, under 1 GB often free).

- Do not load all records into memory unnecessarily. Process **country by country** and in **chunks**.
- Avoid needless pandas copies; free intermediates (`del`, `gc.collect()`).
- Avoid huge dense matrices; use sparse structures. **Never build a full S1 × S2/S3 Cartesian product.**
- Profile memory before large operations. Test on subsets before full runs.
- A simpler solution that runs end-to-end beats a better one that doesn't.

## Competition rules (violations can mean disqualification)
**Use ONLY the supplied dataset.** Do not use external business databases, government registries, commercial entity-resolution APIs, geocoding APIs, internet lookups, external data augmentation, or external identity-resolution services. No pretrained address/business knowledge fetched from the web.

Model constraints: the final model must be **MIT or Apache 2.0 licensed** and **≤ 8B parameters**. Check the license and size before introducing any pretrained model. Classical ML (LightGBM, TF-IDF, rapidfuzz, scikit-learn) is the default; those are permissively licensed.

## Output requirements
The final submission package needs `output/matching_results.tsv` and `output/candidate_pairs.tsv` (working copies are written to `outputs/` in this repo, and copied into `output/` when packaging). Both are tab-separated with the exact headers below.

`matching_results.tsv`, headers `source1_entity_id`, `matched_entity_ids`:
- Exactly one row for every S1 entity in `test_source1.tsv`. No missing or duplicate rows.
- IDs in `matched_entity_ids` are comma-separated, S2- or S3- only (never S1, no self-matches), no duplicates in a list, and must exist in the test set.
- An empty `matched_entity_ids` is valid, and required when there is no match.

`candidate_pairs.tsv`, headers `source1_entity_id`, `candidate_entity_ids`: same rules. Final matches must be a subset of the candidates.

Validate before every submission:
```bash
python data/raw/utils/validate_submission.py -m outputs/matching_results.tsv -c outputs/candidate_pairs.tsv -t data/raw/dataset/test
```
Add `--check-ids` for the ID-existence check (memory-heavy; drop `-c` if it runs out of RAM).

The final zip must also contain `code/business_entity_resolution/` (with `src/`, a README with reproduction steps, and pinned `requirements.txt`) and the filled-in `Documentation_template.md` (template in `data/raw/`). The zip is code-reviewed for external-data use.

## Git and repo rules
- Remote: https://github.com/ahzamhaq/DOOM_WARRIORS (**public**). Never commit competition data, outputs, models, zips or PDFs (`.gitignore` enforces this; keep it that way).
- **Do not add a Claude co-author trailer or any Claude attribution to commits or PR descriptions.** This overrides any default attribution behaviour.
- Commit and push only when asked. Never force-push `main`.
- Do not print or commit tokens, credentials or `.env` files.
- Run everything from the repo root with the venv: `.venv/Scripts/python -m src.predict`.
- Fixed seed lives in `config.py` (`SEED = 42`); keep runs reproducible.
