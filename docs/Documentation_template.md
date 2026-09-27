# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** DOOM_WARRIORS
**Team Members:** Ahzam Haque (lead / integration), Satyam (normalization + blocking), Prashant Roy (features + matching), Vansh (evaluation + submission)
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
The submitted file comes from a fast, precision-first exact-key matcher: an S2/S3 record is matched to an S1 entity
only when, in the same country, the cleaned business name and the first address house number are identical. In
parallel we built and validated a full ML pipeline (normalization → two-stage blocking → 23 pairwise features →
LightGBM with an F0.5-tuned threshold), which reached macro F0.5 0.946 on our held-out development split but could
not be run end to end on the full test set before the deadline.

---

## 2. Methodology

### 2.1 Problem Analysis
- Train: S1 2.21M, S2 5.03M, S3 5.29M; test: S1 1.73M, S2 4.89M, S3 5.08M. Test adds France (not in train), so
  country is treated as an open set and never hard-coded.
- Matches per S1: 5.6 % none, 5.4 % one, 89 % two or more (mean 3.46). Every true pair shares its country; each
  S2/S3 record matches at most one S1; about 26 % of S2/S3 records are decoys that match nothing.
- Noise: junk name prefixes, legal-suffix variants (Pvt/Private, Ltd/Limited, LLC/L.L.C.), website/handle names,
  Indic-script names (Devanagari, Telugu, Tamil, Bengali, …), missing S2/S3 addresses (~3–4 %), casing and
  component reordering, altered house numbers. Postcodes are rare (≈11 % US, ≈1 % India), so not used as keys.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (full pipeline); exact-key rule (submitted file).
**Core Innovation:** precision-first design for macro F0.5: conservative monotone-constrained LightGBM, threshold
tuned for macro F0.5 on held-out S1 entities, and address-number features that reject near-duplicate decoys.

---

## 3. Candidate Generation (Blocking)
- **Submitted file:** exact join on (country, cleaned name, first house number); keys shared by more than 10 pool
  records are dropped. `candidate_pairs.tsv` equals the matched pairs.
- **Full pipeline (`src/blocking.py`):** within each country, stage 1 shortlists 500 records per S1 with word
  unigram+bigram TF-IDF over name and address via a hashed inverted index (very frequent tokens not indexed, so no
  S1 × pool product); stage 2 ranks the shortlist with char 3-gram TF-IDF on name and address and keeps the union of
  the top-20 lists, capped at 40 candidates per S1.
- **Candidate recall (dev world with hard decoys, 30,000 S1):** 0.984 overall (India 0.974, US 0.992) at ≈39.7
  candidates per S1, versus 0.951 for our earlier char-TF-IDF baseline.

---

## 4. Matching Model

**Features used (23):**
- Name: ratio, token-set, token-sort, partial ratio, Jaro-Winkler, core-token Jaccard (legal words removed), char
  TF-IDF cosine; names transliterated to Latin with anyascii so Indic and English spellings compare.
- Address: token-set, ratio, char TF-IDF cosine, house-number agreement, Jaccard and conflict count over all
  address numbers (targets near-duplicate decoys such as 181/1 vs 181/10).
- Other: missing-field indicators, country agreement, script mismatch, domain-style name, S2/S3 source, and the
  blocking scores and ranks.

**Model type:** LightGBM (MIT licence, trained from scratch; no pretrained models or external data), shallow trees,
L1/L2 regularisation, monotone constraints on similarity features.
**Threshold selection method:** macro F0.5 sweep on held-out S1 entities; ties go to the higher threshold.

---

## 5. Results & Error Analysis

- **Submitted exact-key matcher, dev world:** macro F0.5 0.540, pair precision 0.9985, pair recall 0.317.
- **Full ML pipeline, dev world (held-out 6,044 S1):** macro F0.5 0.946 (precision 0.980, recall 0.896), measured
  with a stand-in retriever before the final blocking was merged. Leave-one-country-out: US→India 0.896,
  India→US 0.947.
- **Common false positives:** near-duplicate decoys (same name, one address number changed), common names where
  the candidate has no address.
- **Common false negatives:** altered house numbers, missing addresses, Indic-script and website-style names.

---

## 6. Conclusion
A precision-first exact-key rule gives a safe, fully reproducible submission. The ML pipeline built alongside it
(two-stage blocking with 0.984 candidate recall and an F0.5-tuned LightGBM at 0.946 on held-out data) is the
intended final system; the remaining step was running it end to end on the full test set within the time limit.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: `src/quick_match.py` (produces the submitted outputs:
`python -m src.quick_match test`), and the full pipeline in `src/` (`normalize.py`, `blocking.py`, `features.py`,
`matcher.py`, `evaluate.py`, `predict.py`, `build_dev_world.py`, `config.py`, `data_loader.py`). See its README.

### B. Additional Results
Development data: a deterministic 30,000-S1 slice of the training data (seed 42, 20 % validation split) plus
"hard decoys" (training S2/S3 records sharing a name word) to make precision estimates less optimistic.
