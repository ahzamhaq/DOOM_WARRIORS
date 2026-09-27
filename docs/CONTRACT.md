# Integration contract

Read this before touching any `src/` module. It fixes the **data shapes between modules** so
Persons 2, 3 and 4 can build in parallel and the Lead can wire them together without rewrites.
Anything not written here is private to the module's owner. Changing a shape here needs Lead sign-off
and an update to this file in the same PR.

## 1. Data flow (one country at a time)

```
data_loader ─► normalize ─► blocking ─► features ─► matcher ─► select_matches ─► files
 raw records   +NORM cols    pairs        feats      proba       matches
                              │                        │
                              └─ candidate_pairs.tsv   └─ scored (proba >= floor)
```

`predict.py` owns the loop. For each country found in the data (never a hard-coded list):

1. Load S1 for the country, and S2 + S3 concatenated as the **pool**; normalize both.
2. `index = blocking.build_index(pool)` and `ctx = features.build_context(pool)`, both once per country.
3. For each S1 **chunk** (`CHUNK_S1` rows): `blocking.generate_candidates` → append to `candidate_pairs.tsv`
   → `features.build_features(pairs, chunk, pool, ctx=ctx)` → `model.predict_proba` → keep rows with proba >= `PROBA_FLOOR`.
4. After all chunks: `matcher.select_matches(scored, model.threshold, one_to_one=config.ONE_TO_ONE)` → write `matching_results.tsv`.

Why per country, chunked: RAM (see CLAUDE.md), and every true pair shares a country. Step 4 runs per
country, not per chunk, because the optional one-to-one rule (§6) can only be applied once all of a
country's S1 chunks are scored.

## 2. Shared types

All IDs are strings (`S1-…`, `S2-…`, `S3-…`). Source = ID prefix. Column names are constants in `config.py`.

| Name | Shape | Notes |
| --- | --- | --- |
| **records** | DataFrame, index = `entity_id` (also kept as a column) | columns `entity_id, business_name, business_address, country`; missing = `""`, never NaN |
| **normalized records** | records + `NORM_COLUMNS` | see §3 |
| **pool** | normalized S2 + S3 records of ONE country | source from ID prefix |
| **pairs** | `s1_id, cand_id` + `blk_*` columns | see §4 |
| **feats** | DataFrame, `index == pairs.index`, columns `FEATURE_NAMES`, numeric | see §5 |
| **scored** | `s1_id, cand_id, proba` (float32) | only rows with proba >= `PROBA_FLOOR` |
| **matches** | `s1_id, cand_id` | final; each `cand_id` appears at most once |
| **truth** | `dict[s1_id, set[cand_id]]` | from `evaluate.truth_dict` |

## 3. normalize.py (Person 2)

```python
normalize_records(df) -> DataFrame   # returns df with NORM_COLUMNS added, same index/row order
```
`NORM_COLUMNS` (fixed, in `config.py`; ask the Lead before adding one):

| Column | Type | Meaning |
| --- | --- | --- |
| `name_norm` | str | lowercased, junk/punctuation removed, legal suffixes stripped, `&`→`and` |
| `addr_norm` | str | lowercased, abbreviations standardized, `NULL`/junk tokens removed |
| `name_is_domain` | bool | name is a website/handle (`x.com`, `@x`) |
| `name_script` | str | `latin`, `indic`, or `other` |
| `addr_empty` | bool | address missing or blank |
| `house_no` | str | leading street number if any, else `""` |

Rules: row-wise and stateless (no dependence on other rows, so chunking is safe); deterministic; must not
branch on the country value (France is unseen in training). Raw columns are left untouched.

## 4. blocking.py (Person 2)

```python
build_index(pool) -> BlockIndex                                   # once per country
generate_candidates(s1, index, k=BLOCK_K, max_cands=MAX_CANDS_PER_S1) -> pairs
```
`pairs` columns: `s1_id, cand_id` plus these numeric retrieval features (`BLK_COLUMNS`; NaN / -1 for a
retriever that did not return the pair): `blk_name_score, blk_addr_score, blk_name_rank, blk_addr_rank`.

Invariants (checked by `evaluate.check_pairs`):
- `cand_id` starts with `S2-` or `S3-`, and belongs to the same country as `s1_id`.
- No duplicate `(s1_id, cand_id)`; at most `max_cands` per S1 (bounds memory).
- Only S1 ids from the given chunk. An S1 with no candidates simply has no rows.
- **Identical code and parameters at train and test time**, so `blk_*` distributions match.

The pairs returned are exactly what goes into `candidate_pairs.tsv` and what the matcher scores.

## 5. features.py (Person 3)

```python
FEATURE_NAMES: list[str]                                  # fixed, exported by the module
build_context(pool) -> ctx                                # once per country; opaque to callers
build_features(pairs, s1, pool, ctx=None) -> feats        # s1, pool are normalized records indexed by entity_id
```
**Per-country context (`ctx`).** Corpus statistics (the TF-IDF/IDF weights) are computed once per country by
`build_context(pool)` and passed to every chunk's `build_features` call, instead of being refit per chunk.
- `ctx` depends **only on `pool`** and is deterministic (no labels, no S1 data, no row-order dependence).
- `build_features(..., ctx=None)` computes the same context from `pool` itself, so existing callers keep working.
  **Passing `build_context(pool)` must give identical features to passing `None`** (tested).
- A `ctx` belongs to the one country's pool it was built from: never reuse it across countries or pools.
- Callers treat `ctx` as opaque: no reading or editing its contents outside `features.py`.
- `ctx` is held in RAM for the duration of the country only; it must stay small (no per-record data).
- One row per pair, same index and order as `pairs`. Numeric only (`float32`; NaN allowed, no inf).
- Pass the `blk_*` columns through as features (they are strong signals).
- Depends only on the pair, the two records, and static corpus statistics. **No labels, no `country` string,
  no ID digits.** Derive source (S2/S3) from the ID prefix if useful.
- Corpus statistics (IDF etc.) come from `pool` only, via `ctx` (above).

## 6. matcher.py (Person 3)

```python
class Model:
    feature_names: list[str]
    threshold: float                                   # set after tuning; used by predict.py
    predict_proba(feats) -> np.ndarray                 # shape (n,), P(match); reorders columns by feature_names
    save(path); @classmethod load(path)
train(feats, labels) -> Model                          # labels: int8 Series aligned to feats.index
select_matches(scored, threshold) -> matches
```
`select_matches(scored, threshold, one_to_one=config.ONE_TO_ONE)`:
- Always: keep rows with `proba >= threshold`.
- **`one_to_one=True` is an EXPERIMENTAL option, not an assumption.** In the training ground truth every S2/S3 record
  matched at most one S1, so if a `cand_id` clears the threshold for several S1 entities, this keeps only the
  highest-proba S1 (ties: smallest `s1_id`). That is an observation about training data, not a stated competition
  rule, and test data (including France) may differ. **Default is `False`** (`config.ONE_TO_ONE`).
  Person 3 compares validation macro F0.5 with and without it (also per country, and leave-one-country-out) and
  reports the result; only then does the Lead flip the flag in `config.py`. Both code paths must stay working.
- Returns a DataFrame `s1_id, cand_id`, not a dict, to keep memory low. Model choice stays Person 3's call (LightGBM planned).

## 7. evaluate.py (Person 4)

Already implemented: `f_beta`, `macro_f_beta`, `candidate_recall`, `truth_dict`. To add (signatures fixed):

```python
split_s1(s1_ids, val_frac=0.1, seed=SEED) -> (train_ids, val_ids)   # deterministic, hash-based, S1 level
label_pairs(pairs, truth) -> pd.Series                                # int8 1/0, aligned to pairs.index
check_pairs(pairs, s1, pool) / check_feats(feats, pairs)              # assert the invariants above
blocking_report(pairs, truth, s1_country) -> DataFrame                # recall, avg cands/S1, by country and source
sweep_thresholds(scored, truth, s1_ids, select_fn, grid) -> DataFrame # F0.5 per threshold (select_fn injected: no import cycle)
per_country_report(matches, truth, s1_country) -> DataFrame
```
The dev world (small frozen train subset) is built by the Lead's script, not by `evaluate.py`: see §10.
Scoring always takes the full list of S1 ids: S1 entities with no candidates and true singletons must count.
France has no labels: also report a **leave-one-country-out** score (train on US, validate on India, and the reverse)
as a proxy for how thresholds transfer to an unseen country.

## 8. predict.py (Lead)

```
python -m src.predict train    [--country C] [--limit-s1 N]   # dev world → blocking → features → train → tune threshold → save model
python -m src.predict dev      [--country C] [--limit-s1 N]   # score validation S1, print macro F0.5 + blocking report
python -m src.predict test                                    # full test → outputs/*.tsv
python -m src.predict test --dry-run                          # empty submission (validator smoke test)
```
`run_pipeline(countries, load_country, model, out_dir, comps=REAL, chunk_s1, one_to_one)` is the single integration path
(`comps` = the module functions; tests inject fakes there). `Components.build_feature_context` is optional:
when set, it is called once per country and its result passed to `build_features(..., ctx=...)`; when `None`
(e.g. fakes that predate `ctx`), `build_features` is called without `ctx`. Both output files are written per country as the loop goes (`data_loader.append_id_lists`), so full result
dicts are never held in RAM. Every S1 id gets a row, including those with no candidates or matches.

## 9. Who can start now, and against what

| Person | Can start now using | Blocked on |
| --- | --- | --- |
| 2 | raw records (`load_source(..., country=...)`) | nothing |
| 3 | the frozen dev world (§10): a fake `pairs` frame from its ground truth (positives plus random same-country negatives), `normalize_records` faked as lowercase | nothing; swap in real blocking later |
| 4 | the frozen dev world (§10) and any `pairs` / `scored` frame; the toy test (§11) as a template for `check_pairs` | nothing |
| Lead | stubs raising `NotImplementedError`; `--dry-run` passes the official validator today | teammates' modules for the full run |

## 10. Frozen dev world (`data/processed/dev_world/`, git-ignored)

A small, deterministic slice of the TRAIN data so everyone measures on identical data in seconds.
Build (about 20 s) and load:
```bash
python -m scripts.build_dev_world          # writes data/processed/dev_world/ and compares to docs/dev_world_manifest.json
```
```python
from src.data_loader import load_dev_world
dw = load_dev_world()                      # or load_dev_world("India", split="val")
dw.s1, dw.s2, dw.s3, dw.pool, dw.truth_raw # records indexed by entity_id; s1 has an extra `split` column
```
Contents (seed 42, 30,000 S1): S1 30,000 (US 17,904, India 12,096); split train 23,956 / val 6,044 (20 % val);
S2 117,992 + S3 124,895 pool records; 103,865 positive pairs; 1,651 singletons (5.5 %, like the full 5.6 %).
Of the pool, 103,865 records truly match a sampled S1 and 139,022 are "other" (decoys and records of other entities).
No France (train has none). Files: `s1.tsv, s2.tsv, s3.tsv, ground_truth.tsv` (same formats as raw), `manifest.json`.

Recipe, so anyone can re-derive it (`scripts/build_dev_world.py`; nothing depends on row order or library versions):
1. `u(id)` = splitmix64(numeric part of the id + a seed/source salt) / 2^64, uniform in [0, 1).
2. S1 = the 30,000 train S1 entities with the smallest `u(id)` (natural country mix).
3. Ground truth = their rows of `train_ground_truth.tsv`.
4. Pool = every S2/S3 record that truly matches a sampled S1, plus every other S2/S3 record with `u(id) < f`,
   `f = 30000 / 2,206,821 = 0.0136`. So decoys and other-entity records appear at their natural mix, thinned by `f`.
5. `split` = `val` if `u_split(id) < 0.2` else `train` (frozen; `evaluate.split_s1` should agree with it or defer to it).
6. Sorted by `entity_id`, LF newlines. `docs/dev_world_manifest.json` (committed) holds the sha256 of every file:
   the script prints `MATCHES committed manifest` when your copy is byte-identical.

**Caveat, important:** other-entity confusers are thinned to about 1/73 of their real density, so blocking recall and
matcher precision measured here are **optimistic**. Use the dev world for development, regression tests and relative
comparisons; take final threshold and K choices from a larger run (`--n-s1 300000`, or one full country) before
trusting absolute numbers. It is derived from competition data: never commit it (`data/**` is git-ignored).

## 11. Contract test (`tests/test_contract.py`)

```bash
python -m unittest discover -s tests -t . -v
```
Runs **fake** normalize, blocking, features, model and select_matches through the real `predict.run_pipeline` on a
tiny toy world (4 countries, one unseen; 24 S1; singletons; one S2 record that looks like two S1 entities), then runs
the official `validate_submission.py --check-ids` on the toy output (skipped, with a message, if `data/raw/utils/` is
not unzipped). It checks: one row per S1 in both files, headers, empty rows for no-match S1, matches ⊆ candidates,
S2/S3 ids only, unseen country processed, chunk size does not change the output, `one_to_one` off/on behave as
specified (and is off by default), and the shapes in §2–§6 (columns, index alignment, dtypes). It says nothing about
ML quality. Swapping a fake for the real module must keep it green. It is mutation-checked: it fails when the fake
select ignores `one_to_one`, when blocking drops a candidate, and when an unknown S1 id is emitted.

## 12. Rules of the road
- Import shared names from `config.py`; never re-type column names or paths.
- A module never imports another owner's module, except `predict.py` (integration) and `evaluate.py` receiving callables.
- Seeds come from `config.SEED`. No global mutable state.
- New shared columns or signature changes: PR touching this file, Lead approves.
