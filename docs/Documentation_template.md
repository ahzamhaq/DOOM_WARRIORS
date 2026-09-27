# DOOM_WARRIORS — Business Entity Resolution (Amazon ML Challenge 2026)

[[TBD: move these sections under the headings of the official data/raw/Documentation_template.md (the template
was not available when this draft was written), then delete this line.]]

## 1. Problem and metric

For every Source 1 (S1) business record, find the records in Source 2 and Source 3 (S2/S3) that describe the
same real-world business. An S1 entity can have zero, one or many matches.

The metric is **macro F0.5 over S1 entities**. We compute F0.5 for each S1 entity and average over all of them.
An empty prediction on a true singleton (no matches) scores 1.0, and any match predicted for a singleton scores 0.
With β = 0.5, precision weighs about four times as much as recall, so **a false merge costs more than a missed
match**. This shaped every design choice below: conservative features, a monotone-constrained model and a
threshold chosen for F0.5 on held-out entities.

## 2. Data and exploratory findings

Only the supplied dataset was used: no external databases, registries, APIs, geocoding, web lookups or pretrained
business/address knowledge.

| | S1 | S2 | S3 |
| --- | ---: | ---: | ---: |
| train rows | 2.21 M | 5.03 M | 5.29 M |
| test rows | 1.73 M | 4.89 M | 5.08 M |

- **Test contains France, which is absent from training.** Country is treated as an open set: nothing in the code
  hard-codes, filters on or one-hot encodes a country, and every test S1 entity (France included) is processed.
- Matches per S1: 5.6 % have none, 5.4 % have one, 89 % have two or more (mean 3.46, max 11). S2 and S3 contribute
  about equally.
- Every true match agreed on country. Each S2/S3 record matches at most one S1 entity. About 26 % of S2/S3
  records match nothing (decoys).
- Noise seen: junk prefixes (`--`, `<<`, `#`), legal-suffix variants (Pvt/Private, Ltd/Limited, LLC/L.L.C.,
  SARL/SAS…), typos and lookalike characters, names given as websites or handles, missing S2/S3 addresses (~3–4 %),
  casing and format differences (S2 addresses are upper-case), and Devanagari and other Indic scripts in India
  names.
- Postcodes are unreliable (present in ~11 % of US and ~1 % of India addresses), so they are never a primary key.
- Name-token blocking alone recovered only ~85 % of sampled true pairs. Address-based retrieval recovers
  important misses (other-script names, website-style names).

## 3. Pipeline architecture

```
load ─► normalize ─► blocking (candidates) ─► pair features ─► classifier ─► threshold ─► matches
          per country, S1 in chunks            candidate_pairs.tsv = exactly the pairs scored
```

- The pipeline runs **one country at a time** (every true pair shares a country), with S1 in chunks of 50,000,
  so memory stays bounded (~11.7 M test records on a machine with ~8 GB RAM). It never builds an S1 × S2/S3
  Cartesian product.
- Per country, the retrieval index and the corpus statistics (TF-IDF weights) are built once from that country's
  S2+S3 pool and reused for every chunk. They depend only on the pool, never on labels.
- Training, validation and test prediction all go through **the same chunk loop**, so the model is trained on
  exactly the candidate and feature distributions it sees at test time.
- `candidate_pairs.tsv` is the exact set of pairs fed to the classifier, so every final match is a subset of it.
  Both output files are streamed to disk per country.

## 4. Normalization

A row-wise, stateless, deterministic transformation of names and addresses that never branches on country:

- lower-casing and removal of Latin diacritics and format characters;
- removal of junk prefixes and wrappers, punctuation and `NULL`-type tokens;
- `&` → `and`; legal suffixes standardized and stripped (Pvt/Private, Ltd/Limited, Inc, LLC, PLLC, SARL, SAS…);
- abbreviations standardized (Rd → road, St → street, …);
- flags for website/handle names, the name's script (Latin / Indic / other) and a missing address, plus the
  leading house number.

For comparisons, names and addresses are transliterated to Latin letters with **anyascii** (see §9), so a
Devanagari name and its English spelling can be compared. Without it, 16.9 % of true India pairs lost every name
feature.

[[TBD: final details of Person 2's normalize.py once merged.]]

## 5. Candidate generation (blocking)

Candidates come from within the same country only:

1. character n-gram similarity search on names, top K per S1 (K = 20);
2. character n-gram similarity search on addresses, top K per S1;
3. the union of both lists, capped at 40 candidates per S1.

Each candidate keeps its retrieval scores and ranks (`blk_name_score`, `blk_addr_score`, `blk_name_rank`,
`blk_addr_rank`) as features. Blocking runs with identical code and parameters at training and test time.

Candidate recall (the share of true pairs that reach the classifier, which caps overall recall) is measured by
country and by source with `evaluate.blocking_report`.

| Blocking | Candidate recall | Avg candidates / S1 |
| --- | ---: | ---: |
| Stand-in retriever (char-3-gram TF-IDF, dev world + hard decoys) | 0.9509 | [[TBD]] |
| Final blocking.py | [[TBD]] | [[TBD]] |

## 6. Pair features (21, all country-agnostic)

| Group | Features |
| --- | --- |
| Name | `name_ratio`, `name_token_set`, `name_token_sort`, `name_partial`, `name_jaro_winkler` (rapidfuzz), `name_core_jaccard` (tokens without legal/filler words), `name_tfidf` (char-3-gram TF-IDF cosine) |
| Address | `addr_token_set`, `addr_ratio`, `addr_tfidf` |
| House number | `house_num`: 1 same number, 0 unknown, −1 both present but different |
| Missing fields | `name_missing`, `addr_missing` (number of sides with an empty field) |
| Consistency | `country_match` (agreement only; the country value is never a feature), `script_mismatch`, `domain_name`, `cand_is_s3` |
| Retrieval | `blk_name_score`, `blk_addr_score`, `blk_name_rank`, `blk_addr_rank` |

- Features depend only on the pair, the two records and statistics of the country's own pool. They use no
  labels, no country string and no ID digits.
- TF-IDF uses stateless char-3-gram hashing (2^20 buckets), with IDF fitted on a deterministic sample of the
  pool.
- Every distinct record is processed once rather than once per pair (34–59 µs per pair, ~0.5 GB per
  50,000-S1 chunk).

## 7. Matching model

- **LightGBM** binary classifier (MIT licence, well under 8 B parameters) on the 21 features, trained only on
  labels from the training ground truth.
- Kept conservative for a precision-heavy metric:
  - shallow trees (max depth 6, 31 leaves, ≥ 200 samples per leaf);
  - L1 = 1, L2 = 5 regularization, with row and column subsampling at 0.8;
  - learning rate 0.05, up to 400 trees with early stopping on a held-out subset of the training S1.
- **Monotone constraints**: a higher name or address similarity, house-number agreement, country agreement or
  retrieval score can never lower P(match). This keeps behaviour sensible on the unseen country.
- Seed 42 everywhere; the run is reproducible.

## 8. Threshold and match selection

- A pair is a match when P(match) ≥ threshold.
- The threshold is chosen on held-out validation S1 entities from the grid 0.10, 0.15, …, 0.95. We take the one
  with the highest macro F0.5; **ties go to the highest threshold** (precision first).
- **The one-to-one constraint is off.** In the training ground truth, each S2/S3 record matched at most one S1.
  Enforcing that at prediction time changed validation macro F0.5 by only +0.0007 / ±0.000, and it is not a
  stated competition rule (test data, especially France, may differ). The option stays in the code, off by
  default.

## 9. Validation protocol

- **S1-level split.** All pairs of an S1 entity sit on the same side, so no validation entity influences
  training. The frozen development set (30,000 train S1, seed 42) uses a fixed 80/20 hash split. Larger runs use
  a deterministic splitmix64 hash of the ID, which gives the same split on every machine.
- **Scoring always covers every S1 entity**, including those with no candidates and true singletons, exactly as
  the competition metric does.
- **Leave-one-country-out.** France has no labels, so we train on one country and validate on the other (US →
  India and India → US). This estimates how the model and threshold transfer to an unseen country.
- **Contract checks** run on every candidate set and feature matrix. Candidates must be S2/S3 IDs of the same
  country, with no duplicates and at most 40 per S1. Features must be aligned to the pairs, float32 and free of
  inf values.
- **Decoy caveat.** The development subset thins other-entity confusers, so it was topped up with "hard decoys"
  (training records sharing a name word with a sampled S1). Final numbers come from [[TBD: larger run / full
  country]].

## 10. Results

| Setting | Candidate recall | Precision | Recall | Macro F0.5 | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dev world + hard decoys, stand-in blocking | 0.9509 | 0.9757 | 0.8902 | 0.9400 | 0.70 |
| Final pipeline, validation | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] | [[TBD]] |
| Leaderboard | | | | [[TBD]] | |

Per country (stand-in blocking): US 0.957, India 0.915. Leave-one-country-out: US → India 0.889,
India → US 0.942. [[TBD: final per-country and LOCO numbers from `predict train` / `notebooks.check_evaluate`.]]

## 11. Compliance

- **Data.** Only the competition dataset. No external data, APIs, internet lookups or augmentation; the code
  makes no network calls (checked automatically when the package is built).
- **Model.** LightGBM (MIT), trained from scratch; no pretrained model.
- **Libraries** (pinned in `requirements.txt`): pandas, numpy, pyarrow, scikit-learn, lightgbm, rapidfuzz, tqdm
  (BSD/MIT/Apache-2.0).
- **anyascii 0.3.3** (ISC licence) provides generic Unicode-to-ASCII transliteration tables. It contains no
  business, address or entity data.

## 12. Reproduction

See `code/business_entity_resolution/README.md`. Both output files pass the official validator
(`validate_submission.py --check-ids`) before the package is built.

## 13. Team

| Member | Role |
| --- | --- |
| Person 1 | Lead: integration, pipeline driver, final runs |
| Person 2 | Normalization and blocking |
| Person 3 | Features and matching model |
| Person 4 | Evaluation, validation, submission package and documentation |

[[TBD: team member names.]]
