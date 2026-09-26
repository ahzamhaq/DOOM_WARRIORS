"""Competition metric: macro F0.5 over Source 1 entities (singletons included).

Owner: Person 4. Also home for the validation split and submission checks.
Contract: docs/CONTRACT.md §7. Implemented: metric + truth parsing. Still to implement (signatures fixed):
split_s1, label_pairs, check_pairs, check_feats, blocking_report, sweep_thresholds,
per_country_report  (stubs at the bottom of this file).
"""
import pandas as pd

from src.config import BETA, SEED


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
# To implement (Person 4). Signatures are the integration contract.
# ---------------------------------------------------------------------------
def split_s1(s1_ids, val_frac: float = 0.1, seed: int = SEED):
    """Deterministic, hash-based S1-level split -> (train_ids, val_ids). Same result on every machine."""
    raise NotImplementedError


def label_pairs(pairs: pd.DataFrame, truth: dict) -> pd.Series:
    """int8 1/0 per pair (is cand_id in truth[s1_id]), aligned to pairs.index."""
    raise NotImplementedError


def check_pairs(pairs: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame) -> None:
    """Assert blocking invariants (CONTRACT §4): S2/S3 ids only, same country, no duplicates, cap per S1."""
    raise NotImplementedError


def check_feats(feats: pd.DataFrame, pairs: pd.DataFrame) -> None:
    """Assert feature invariants (CONTRACT §5): index == pairs.index, numeric, no inf, fixed columns."""
    raise NotImplementedError


def blocking_report(pairs: pd.DataFrame, truth: dict, s1_country: pd.Series) -> pd.DataFrame:
    """Candidate recall and avg candidates per S1, by country and by source (S2/S3)."""
    raise NotImplementedError


def sweep_thresholds(scored: pd.DataFrame, truth: dict, s1_ids, select_fn, grid) -> pd.DataFrame:
    """Macro F0.5 (+ precision/recall) per threshold. `select_fn(scored, t)` is matcher.select_matches,
    injected by the caller so this module never imports matcher. `s1_ids` = ALL S1 ids scored."""
    raise NotImplementedError


def per_country_report(matches: pd.DataFrame, truth: dict, s1_country: pd.Series) -> pd.DataFrame:
    """Macro F0.5 by country, plus leave-one-country-out numbers as a proxy for unseen France."""
    raise NotImplementedError
