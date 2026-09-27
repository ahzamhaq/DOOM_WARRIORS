"""Competition metric: macro F0.5 over Source 1 entities (singletons included).

Owner: Person 4. Also home for the validation split and submission checks.
Contract: docs/CONTRACT.md §7.

    f_beta, macro_f_beta, candidate_recall, truth_dict      metric + ground truth
    split_s1                                                 deterministic S1-level split
    label_pairs                                              1/0 per candidate pair
    check_pairs, check_feats                                 contract invariants (§4, §5); raise AssertionError
    blocking_report                                          candidate recall + load, by country and source
    sweep_thresholds, per_country_report, loco_report        official validation scoring

Never imports another owner's module: selection / scoring callables are injected by the caller.
All scoring runs over the FULL list of S1 ids, so S1 with no candidates and true singletons count.
"""
import numpy as np
import pandas as pd

from src.config import BETA, BLK_COLUMNS, CAND_ID, MAX_CANDS_PER_S1, S1_ID, SEED

POOL_SOURCES = ("S2", "S3")
_SPLIT_SALT = 5  # scripts/build_dev_world.py uses salts 1-4 (4 = frozen dev split); keep split_s1 independent of them


def f_beta(pred: set, truth: set, beta: float = BETA) -> float:
    """Per-entity F-beta. Empty prediction on a true singleton scores 1.0."""
    if not truth and not pred:
        return 1.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_f_beta(pred: dict, truth: dict, beta: float = BETA) -> float:
    """Average F-beta over every S1 id in `truth` (missing predictions count as empty)."""
    if not truth:
        return 0.0
    return sum(f_beta(set(pred.get(s1, ())), set(t), beta) for s1, t in truth.items()) / len(truth)


def candidate_recall(candidates: dict, truth: dict) -> float:
    """Fraction of true pairs present in the candidate set (blocking recall ceiling)."""
    total = sum(len(t) for t in truth.values())
    hit = sum(len(set(t) & set(candidates.get(s1, ()))) for s1, t in truth.items())
    return hit / total if total else 1.0


def truth_dict(gt_raw) -> dict:
    """Ground-truth DataFrame (raw format) -> {s1_id: set(match_ids)}."""
    return {
        s1: set(m.split(",")) if m else set()
        for s1, m in zip(gt_raw["source1_entity_id"], gt_raw["matched_entity_ids"].astype(str))
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _u01(ids, salt: int, seed: int = SEED) -> np.ndarray:
    """Uniform [0, 1) per id: splitmix64 of the numeric id part. Same recipe as scripts/build_dev_world.u01,
    so it is identical on every machine and independent of row order and library versions."""
    try:
        num = np.fromiter((int(s[3:]) for s in ids), dtype=np.uint64)
    except ValueError as exc:
        raise ValueError(f"ids must look like 'S1-<digits>': {exc}") from None
    mix = (seed * 1_000_003 + salt) * 0x9E3779B97F4A7C15 % 2**64
    with np.errstate(over="ignore"):
        x = num + np.uint64(mix)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return x.astype(np.float64) / 2.0**64


def _obj(values) -> np.ndarray:
    """Any id column (Arrow / str / object) -> object ndarray, so hashing and comparisons behave the same."""
    return np.asarray(pd.Series(values, copy=False).astype(object), dtype=object)


def _true_arrays(truth: dict, s1_ids) -> tuple[np.ndarray, np.ndarray]:
    """(s1_id, match_id) arrays of the true pairs of the given S1 ids."""
    s, c = [], []
    for sid in s1_ids:
        ms = truth.get(sid)
        if ms:
            s.extend([sid] * len(ms))
            c.extend(ms)
    return np.array(s, dtype=object), np.array(c, dtype=object)


def _is_true(s1, cand, truth: dict, true_pairs: tuple | None = None) -> np.ndarray:
    """Boolean array: is (s1[i], cand[i]) a true pair. Strings are factorized to one int64 key per pair,
    then a hash lookup: no Python loop per pair (millions of pairs in seconds)."""
    s1, cand = _obj(s1), _obj(cand)
    n = len(s1)
    ts, tc = _true_arrays(truth, pd.unique(s1)) if true_pairs is None else true_pairs
    if n == 0 or len(ts) == 0:
        return np.zeros(n, dtype=bool)
    sc, _ = pd.factorize(np.concatenate([s1, ts]))
    cc, cu = pd.factorize(np.concatenate([cand, tc]))
    key = sc.astype(np.int64) * len(cu) + cc
    return pd.Series(key[:n]).isin(key[n:]).to_numpy() & (sc[:n] >= 0) & (cc[:n] >= 0)


def _examples(values, mask, k: int = 3) -> str:
    return ", ".join(map(str, pd.Series(np.asarray(values, dtype=object)[np.asarray(mask)]).head(k)))


def _scoring_ctx(truth: dict, s1_ids) -> tuple:
    """Per-universe data reused across thresholds: (ids Index, true count per id, true pair arrays)."""
    ids = pd.Index(pd.unique(_obj(list(s1_ids))), dtype=object)
    n_true = np.fromiter((len(truth.get(s, ())) for s in ids), dtype=np.int64, count=len(ids))
    return ids, n_true, _true_arrays(truth, ids)


def _score_matches(matches: pd.DataFrame, truth: dict, s1_ids, beta: float = BETA, ctx: tuple | None = None) -> dict:
    """Macro F-beta over ALL `s1_ids` (predictions for other S1 are ignored) + pair precision/recall.

    Reproduces notebooks/train_matcher.score exactly (Person 3's validation numbers), vectorized.
    """
    ids, n_true, true_pairs = ctx or _scoring_ctx(truth, s1_ids)
    n = len(ids)
    m = matches[[S1_ID, CAND_ID]].drop_duplicates()
    sid, cid = _obj(m[S1_ID]), _obj(m[CAND_ID])
    pos = ids.get_indexer(sid)
    keep = pos >= 0
    hit = _is_true(sid[keep], cid[keep], truth, true_pairs)
    n_pred = np.bincount(pos[keep], minlength=n).astype(np.int64)
    tp = np.bincount(pos[keep], weights=hit, minlength=n).astype(np.int64)

    b2 = beta * beta
    f = np.zeros(n, dtype=np.float64)
    f[(n_true == 0) & (n_pred == 0)] = 1.0
    ok = tp > 0
    p = tp[ok] / n_pred[ok]
    r = tp[ok] / n_true[ok]
    f[ok] = (1 + b2) * p * r / (b2 * p + r)
    single = n_true == 0
    TP, P, T = int(tp.sum()), int(n_pred.sum()), int(n_true.sum())
    return {
        "n_s1": n,
        "precision": TP / P if P else 1.0,
        "recall": TP / T if T else 1.0,
        "macro_f05": float(f.sum() / n) if n else 0.0,
        "singleton_acc": float((n_pred[single] == 0).mean()) if single.any() else 1.0,
        "pred_pairs": P,
        "true_pairs": T,
    }


def _best_row(sweep: pd.DataFrame) -> pd.Series:
    """CONTRACT §7 threshold rule: highest macro_f05; ties -> highest threshold (precision first)."""
    top = sweep["macro_f05"].max()
    return sweep[sweep["macro_f05"] >= top - 1e-12].sort_values("threshold").iloc[-1]


# ---------------------------------------------------------------------------
# CONTRACT §7 (signatures fixed)
# ---------------------------------------------------------------------------
def split_s1(s1_ids, val_frac: float = 0.1, seed: int = SEED):
    """Deterministic, hash-based S1-level split -> (train_ids, val_ids). Same result on every machine.

    * `s1_ids` = the frozen dev-world S1 frame (has a `split` column, CONTRACT §10): that column IS the split
      and is returned as-is (`val_frac` / `seed` ignored), so everyone measures on the same validation S1.
    * `s1_ids` = ids (list / Index / Series): an id goes to validation iff u(id) < val_frac, where u is the
      splitmix64 hash of scripts/build_dev_world.py with its own salt. Membership depends only on the id and
      seed (not on row order, the other ids or the machine), and a smaller `val_frac` gives a subset.
      The salt differs from the dev world's frozen split on purpose: carving an early-stopping set out of
      the dev TRAIN split with the frozen salt would come back empty (every train id has u >= 0.2).
    Both lists keep the input order; they are disjoint and together equal the input.
    """
    if isinstance(s1_ids, pd.DataFrame):
        if "split" not in s1_ids.columns:
            raise ValueError("split_s1: a DataFrame argument must carry the frozen dev-world `split` column")
        bad = ~s1_ids["split"].isin(["train", "val"])
        if bad.any():
            raise ValueError(f"split_s1: unexpected `split` values {sorted(set(s1_ids['split'][bad]))[:5]}")
        ids = _obj(s1_ids.index)
        is_val = (s1_ids["split"] == "val").to_numpy(dtype=bool)
    else:
        if not 0.0 <= val_frac <= 1.0:
            raise ValueError(f"split_s1: val_frac must be in [0, 1], got {val_frac}")
        ids = _obj(s1_ids.to_numpy() if isinstance(s1_ids, (pd.Series, pd.Index)) else list(s1_ids))
        is_val = _u01(ids, _SPLIT_SALT, seed) < val_frac
    return ids[~is_val].tolist(), ids[is_val].tolist()


def label_pairs(pairs: pd.DataFrame, truth: dict) -> pd.Series:
    """int8 1/0 per pair (is cand_id in truth[s1_id]), aligned to pairs.index."""
    return pd.Series(_is_true(pairs[S1_ID], pairs[CAND_ID], truth).astype(np.int8), index=pairs.index,
                     name="label")


def check_pairs(pairs: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame,
                max_cands: int = MAX_CANDS_PER_S1) -> None:
    """Assert blocking invariants (CONTRACT §4): S2/S3 ids only, same country, no duplicates, cap per S1.

    `s1` = the chunk given to generate_candidates, `pool` = that country's S2+S3 (both indexed by entity_id).
    Collects every violation, then raises one AssertionError listing them (with example ids). Not `assert`:
    it still runs under `python -O`.
    """
    missing = [c for c in (S1_ID, CAND_ID, *BLK_COLUMNS) if c not in pairs.columns]
    if missing:
        raise AssertionError(f"check_pairs: pairs is missing columns {missing}")
    problems = []
    sid, cid = _obj(pairs[S1_ID]), _obj(pairs[CAND_ID])

    if not pairs.index.is_unique:
        problems.append("pairs.index has duplicate labels (features must align to it one-to-one)")
    null = pd.isna(sid) | pd.isna(cid)
    if null.any():
        problems.append(f"{int(null.sum()):,} pairs with a null id")
    bad_src = ~pd.Series(cid, dtype=object).astype(str).str.startswith(("S2-", "S3-")).to_numpy(dtype=bool)
    if bad_src.any():
        problems.append(f"{int(bad_src.sum()):,} cand_id not S2-/S3- (e.g. {_examples(cid, bad_src)})")

    if not s1.index.is_unique or not pool.index.is_unique:
        raise AssertionError("check_pairs failed (CONTRACT §4):\n  - s1 / pool index (entity_id) is not unique"
                             + "".join(f"\n  - {p}" for p in problems))
    # object Index + get_indexer: one hash lookup per pair (isin against an Arrow index is ~10x slower)
    pos1 = pd.Index(_obj(s1.index), dtype=object).get_indexer(sid)
    pos2 = pd.Index(_obj(pool.index), dtype=object).get_indexer(cid)
    not_chunk, not_pool = pos1 < 0, pos2 < 0
    if not_chunk.any():
        problems.append(f"{int(not_chunk.sum()):,} pairs whose s1_id is not in the given S1 chunk "
                        f"(e.g. {_examples(sid, not_chunk)})")
    if not_pool.any():
        problems.append(f"{int(not_pool.sum()):,} pairs whose cand_id is not in the pool "
                        f"(e.g. {_examples(cid, not_pool)})")
    known = ~not_chunk & ~not_pool
    diff = np.zeros(len(sid), dtype=bool)
    diff[known] = _obj(s1["country"])[pos1[known]] != _obj(pool["country"])[pos2[known]]
    if diff.any():
        problems.append(f"{int(diff.sum()):,} pairs across countries (e.g. "
                        f"{_examples(sid + '~' + cid, diff)})")

    dup = pairs.duplicated([S1_ID, CAND_ID]).to_numpy()
    if dup.any():
        problems.append(f"{int(dup.sum()):,} duplicate (s1_id, cand_id) pairs (e.g. {_examples(sid, dup)})")
    per = pd.Series(sid).value_counts()
    over = per[per > max_cands]
    if len(over):
        problems.append(f"{len(over):,} S1 with more than {max_cands} candidates "
                        f"(max {int(over.max())}, e.g. {', '.join(map(str, over.index[:3]))})")

    for c in BLK_COLUMNS:
        if not pd.api.types.is_numeric_dtype(pairs[c]) or pd.api.types.is_bool_dtype(pairs[c]):
            problems.append(f"{c} is not numeric (dtype {pairs[c].dtype})")
        elif np.isinf(pairs[c].to_numpy(dtype=np.float64, na_value=np.nan)).any():
            problems.append(f"{c} contains inf")

    if problems:
        raise AssertionError("check_pairs failed (CONTRACT §4):\n  - " + "\n  - ".join(problems))


def check_feats(feats: pd.DataFrame, pairs: pd.DataFrame, feature_names=None) -> None:
    """Assert feature invariants (CONTRACT §5): index == pairs.index, numeric, no inf, fixed columns.

    `feature_names`: pass features.FEATURE_NAMES to also check the exact column order (the caller passes
    it so this module never imports features.py). Values: float32, NaN allowed, no inf.
    """
    problems = []
    if len(feats) != len(pairs):
        problems.append(f"{len(feats):,} feature rows for {len(pairs):,} pairs")
    if not feats.index.equals(pairs.index):
        problems.append("feats.index != pairs.index (same labels in the same order required)")
    if feats.columns.duplicated().any():
        problems.append(f"duplicate columns {list(feats.columns[feats.columns.duplicated()])}")
    if feature_names is not None and list(feats.columns) != list(feature_names):
        extra = [c for c in feats.columns if c not in feature_names]
        lost = [c for c in feature_names if c not in feats.columns]
        problems.append(f"columns differ from FEATURE_NAMES (missing {lost}, unexpected {extra}"
                        + (", or wrong order)" if not extra and not lost else ")"))
    for c in feats.columns:
        col = feats[c]
        if isinstance(col, pd.DataFrame):
            continue  # duplicate column name, already reported
        if col.dtype != np.float32:
            problems.append(f"{c} has dtype {col.dtype}, expected float32")
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            n_inf = int(np.isinf(col.to_numpy(dtype=np.float64, na_value=np.nan)).sum())
            if n_inf:
                problems.append(f"{c} has {n_inf:,} inf values")
    if problems:
        raise AssertionError("check_feats failed (CONTRACT §5):\n  - " + "\n  - ".join(problems))


def blocking_report(pairs: pd.DataFrame, truth: dict, s1_country: pd.Series) -> pd.DataFrame:
    """Candidate recall and avg candidates per S1, by country and by source (S2/S3).

    `s1_country`: country per S1 id (index = s1_id); it defines the S1 universe (pass ALL S1 of the run,
    including those without candidates). Pairs of other S1 are ignored; duplicate pairs count once.
    One row per (country, source) for every country (sorted) and `ALL`; source `all` = S2 + S3. Columns:
      n_s1              S1 entities in the group
      true_pairs        true (s1, cand) pairs of those S1 from that source
      found_pairs       of which in the candidates
      recall            found_pairs / true_pairs (NaN when there are no true pairs); the matcher's ceiling
      s1_all_found      share of S1 with >= 1 true pair (from that source) whose true pairs are ALL candidates
      avg_cands_per_s1  candidates / n_s1 (S1 without candidates count as 0)
      max_cands_per_s1  largest candidate list
      s1_no_cands       share of S1 with no candidate (from that source)
    """
    s1_country = s1_country[~s1_country.index.duplicated()]
    universe = pd.Index(_obj(s1_country.index), dtype=object)
    country = _obj(s1_country.to_numpy())
    n_all, k = len(universe), len(POOL_SOURCES) + 1  # source codes: S2, S3, then "other" (check_pairs flags it)

    def src_code(ids: np.ndarray) -> np.ndarray:
        return pd.Series(ids, dtype=object).str[:2].map({s: i for i, s in enumerate(POOL_SOURCES)}) \
            .fillna(k - 1).to_numpy(np.int64)

    def per_s1(pos: np.ndarray, src: np.ndarray, weights=None) -> np.ndarray:
        """-> (n_S1, k) counts per universe S1 and source."""
        flat = np.bincount(pos * k + src, weights=weights, minlength=n_all * k)
        return flat.astype(np.int64).reshape(n_all, k)

    p = pairs[[S1_ID, CAND_ID]].drop_duplicates()
    sid, cid = _obj(p[S1_ID]), _obj(p[CAND_ID])
    pos = universe.get_indexer(sid)
    keep = pos >= 0
    sid, cid, pos = sid[keep], cid[keep], pos[keep]
    src = src_code(cid)
    cands = per_s1(pos, src)
    found = per_s1(pos, src, _is_true(sid, cid, truth))
    ts, tc = _true_arrays(truth, universe)
    true = per_s1(universe.get_indexer(ts), src_code(tc))
    stats = {"all": (cands.sum(1), found.sum(1), true.sum(1))}
    stats.update({s: (cands[:, i], found[:, i], true[:, i]) for i, s in enumerate(POOL_SOURCES)})

    rows = []
    for c in [*sorted(set(country)), "ALL"]:
        mask = np.ones(n_all, dtype=bool) if c == "ALL" else country == c
        n = int(mask.sum())
        for src_name, arrays in stats.items():
            cands, found, true = (a[mask] for a in arrays)
            has = true > 0
            rows.append({
                "country": c, "source": src_name, "n_s1": n,
                "true_pairs": int(true.sum()), "found_pairs": int(found.sum()),
                "recall": found.sum() / true.sum() if true.sum() else np.nan,
                "s1_all_found": float((found[has] == true[has]).mean()) if has.any() else np.nan,
                "avg_cands_per_s1": cands.sum() / n if n else np.nan,
                "max_cands_per_s1": int(cands.max()) if n else 0,
                "s1_no_cands": float((cands == 0).mean()) if n else np.nan,
            })
    return pd.DataFrame(rows)


def sweep_thresholds(scored: pd.DataFrame, truth: dict, s1_ids, select_fn, grid) -> pd.DataFrame:
    """Macro F0.5 (+ precision/recall) per threshold. `select_fn(scored, t)` is matcher.select_matches,
    injected by the caller so this module never imports matcher. `s1_ids` = ALL S1 ids scored.

    One row per grid value (in grid order): threshold, precision, recall, macro_f05, singleton_acc,
    pred_pairs, true_pairs, n_s1. Same numbers as notebooks/train_matcher.sweep.
    Pick the threshold with the §7 rule (highest macro_f05, ties -> highest threshold).
    """
    ctx = _scoring_ctx(truth, s1_ids)
    rows = [{"threshold": float(t), **_score_matches(select_fn(scored, t), truth, s1_ids, ctx=ctx)} for t in grid]
    cols = ["threshold", "precision", "recall", "macro_f05", "singleton_acc", "pred_pairs", "true_pairs", "n_s1"]
    return pd.DataFrame(rows, columns=cols)


def per_country_report(matches: pd.DataFrame, truth: dict, s1_country: pd.Series) -> pd.DataFrame:
    """Macro F0.5 by country, plus leave-one-country-out numbers as a proxy for unseen France.

    `s1_country` (index = s1_id) defines the S1 scored: every one counts, with or without matches.
    One row per country (sorted) and `ALL`: country, n_s1, precision, recall, macro_f05, singleton_acc,
    pred_pairs, true_pairs. `ALL` is the n_s1-weighted mean of the countries.
    The leave-one-country-out numbers need a model trained without the held-out country, which only the
    caller can train: run `loco_report` with that model's scores and show both tables.
    """
    s1_country = s1_country[~s1_country.index.duplicated()]
    rows = []
    for c in [*sorted(set(_obj(s1_country.to_numpy()))), "ALL"]:
        ids = s1_country.index if c == "ALL" else s1_country.index[_obj(s1_country.to_numpy()) == c]
        rows.append({"country": c, **_score_matches(matches, truth, ids)})
    cols = ["country", "n_s1", "precision", "recall", "macro_f05", "singleton_acc", "pred_pairs", "true_pairs"]
    return pd.DataFrame(rows, columns=cols)


def loco_report(score_held_out, truth: dict, s1_country: pd.Series, select_fn, grid,
                threshold: float) -> pd.DataFrame:
    """Leave-one-country-out: how a threshold tuned elsewhere transfers to a country the model never saw
    (the proxy for France, CONTRACT §7).

    `score_held_out(country) -> scored` (s1_id, cand_id, proba) for that country's S1, from a model
    trained WITHOUT it (the caller trains; this module never imports matcher). `s1_country` = the S1
    to score (index = s1_id). One row per held-out country: train_on, validate_on, n_s1,
    f05_at_threshold (at the main model's `threshold`), best_threshold_there, best_f05_there
    (what tuning on that country itself would give; the gap is the transfer cost).
    """
    s1_country = s1_country[~s1_country.index.duplicated()]
    country = _obj(s1_country.to_numpy())
    countries = sorted(set(country))
    rows = []
    for held in countries:
        ids = s1_country.index[country == held]
        scored = score_held_out(held)
        sweep = sweep_thresholds(scored, truth, ids, select_fn, grid)
        at = _score_matches(select_fn(scored, threshold), truth, ids)
        best = _best_row(sweep)
        rows.append({"train_on": "+".join(c for c in countries if c != held), "validate_on": held,
                     "n_s1": len(ids), "f05_at_threshold": at["macro_f05"], "threshold": float(threshold),
                     "best_threshold_there": float(best["threshold"]), "best_f05_there": float(best["macro_f05"])})
    return pd.DataFrame(rows)
