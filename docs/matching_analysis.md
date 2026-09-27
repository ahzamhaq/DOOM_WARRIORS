# Matching analysis (Person 3)

Frozen dev world + hard decoys, frozen split (6,044 validation S1), **stand-in blocking** (blocking.py
not merged yet). Aggregates only: no competition records in this file (repo is public).
Reproduce: `python -m notebooks.train_matcher` then `python -m notebooks.error_analysis`.

## Result

| | macro F0.5 | precision | recall | false merges | LOCO US→India | LOCO India→US |
|---|---|---|---|---|---|---|
| 21 features (merged PR #1) | 0.9409 | 0.9758 | 0.8914 | 463 | 0.8922 | 0.9427 |
| **+ 2 address-number features (this PR)** | **0.9467** | **0.9827** | 0.8903 | **328 (-29 %)** | **0.8982** | **0.9479** |

Threshold 0.75 (best on validation). One-to-one: +0.0006, keep it off. `build_features`: 458k India
pairs in 16 s (35 µs/pair), peak +0.35 GB. Context (PR #2) unchanged.

Feature ablation (best-threshold macro F0.5):

| features | val | LOCO→India | LOCO→US | kept? |
|---|---|---|---|---|
| 21 (before) | 0.9409 | 0.8922 | 0.9427 | |
| + address numbers | 0.9467 | 0.8982 | 0.9479 | **yes**: better everywhere |
| + name extra-token count | 0.9477 | 0.8822 | 0.9494 | no: hurts India LOCO |
| + pool name frequency | 0.9504 | 0.8815 | 0.9555 | no: hurts India LOCO, and depends on pool density (dev world thinned vs full test pools) |

## 1. False merges (threshold 0.75): 328 of 18,971 predicted pairs

| type (first rule that matches) | count | share | India | US | F0.5 if all fixed |
|---|---|---|---|---|---|
| same name and address (near-duplicate) | 105 | 32 % | 82 | 23 | 0.9508 |
| same name, different/missing address | 82 | 25 % | 32 | 50 | 0.9501 |
| other | 79 | 24 % | 67 | 12 | 0.9500 |
| house number conflict | 27 | 8 % | 11 | 16 | 0.9477 |
| same address, different name | 25 | 8 % | 14 | 11 | 0.9476 |
| script mismatch (Indic vs Latin) | 9 | 3 % | 9 | 0 | 0.9470 |
| domain-style name | 1 | 0 % | 0 | 1 | 0.9468 |

- **Near-duplicates** are almost all decoys (104/105 match no S1): a copy of the true record with a
  word added to the name ("... Exports", "... Group") and a number in the address changed
  (e.g. 181/1 → 181/10). The two new features target exactly this (count was 147 before).
- **Same name, different/missing address**: 80/82 belong to *another* S1, usually a candidate with
  no address and a common name. Hard to fix at the matcher; a name-frequency feature helped
  in-country but did not transfer (see ablation).
- **Same address, different name** is small (8 %), so an address × name interaction is not justified.
- 20 false merges hit true singletons (17 S1), each costing that S1 the full 1.0.

## 2. Missed matches: 2,298 of 20,941 true pairs

| type | blocking (never a candidate) | matcher (below threshold) | share of all true |
|---|---|---|---|
| altered house number | 110 | 531 | 3.1 % |
| other | 302 | 234 | 2.6 % |
| missing address | 236 | 276 | 2.4 % |
| **Indian-script name** | **289** | 42 | 1.6 % |
| name very different | 24 | 153 | 0.9 % |
| **website/handle name** | **82** | 14 | 0.5 % |

By country/source (blocking / matcher misses): India S2 277/217, India S3 413/210, US S2 160/482, US S3 197/342.
**For Person 2:** Indian-script and website names are mostly *blocking* misses (the right record never
reaches the matcher); India S3 is the weakest slice. Altered house numbers are mostly the matcher's
(the conservative threshold's price).

## 3. Blocking baseline for Person 2 (stand-in, all 30,000 dev S1)

Country-level char-trigram TF-IDF, name and address retrievers, top-20 each, union capped at 40.

| country | source | true pairs | recall | name retriever only | address retriever only | cands/S1 |
|---|---|---|---|---|---|---|
| India | S2 | 20,542 | 0.934 | 0.525 | 0.881 | 37.9 |
| India | S3 | 21,533 | 0.908 | 0.530 | 0.783 | 37.9 |
| US | S2 | 30,061 | 0.975 | 0.731 | 0.897 | 37.5 |
| US | S3 | 31,729 | 0.968 | 0.700 | 0.884 | 37.5 |
| **all** | | 103,865 | **0.951** | 0.639 | 0.866 | 37.6 |

Target: blocking.py should reach ≥ 0.951 overall at ≤ 40 candidates per S1. The name retriever is
weak on India (0.53): transliterating Indic names before retrieval should help.
`python -m notebooks.train_matcher --blocking standin|real` compares the two on the same data.

## 4. Retraining is one command

`python -m notebooks.train_matcher` uses the real `normalize_records` and `blocking.build_index` +
`generate_candidates` automatically once implemented (stand-in only on `NotImplementedError`;
`--blocking real` fails loudly instead). The report includes the threshold sweep, one-to-one on/off,
per-country, LOCO, the France rule and blocking recall. It saves `outputs/model.joblib` with the threshold.

## 5. Threshold for France (no labels)

Rule: **use the mean of the leave-one-country-out best thresholds**, snapped to the 0.05 grid.
Models applied to a country they were not trained on need a *lower* threshold than in-country
validation suggests (LOCO bests 0.55 / 0.60 vs 0.75 pooled).

| rule | threshold | worst regret | India | US |
|---|---|---|---|---|
| **mean of LOCO bests** | **0.55** | **0.0007** | 0.0000 | 0.0007 |
| stricter LOCO best | 0.60 | 0.0010 | 0.0010 | 0.0000 |
| pooled validation best | 0.75 | 0.0038 | 0.0023 | 0.0038 |

Proposal: US/India 0.75 (validated), **France 0.55–0.60** (0.60 if we prefer precision; costs ≤ 0.001).
Needs a per-country threshold in predict.py (Lead's call). Recompute once real blocking lands.
