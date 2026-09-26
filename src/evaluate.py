"""Competition metric: macro F0.5 over Source 1 entities (singletons included).

Owner: Person 4. Also home for the validation split and submission checks.
"""
from src.config import BETA


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
