"""Pair classifier + threshold -> final matches. Owner: Person 3.

LightGBM binary classifier on features.py output, trained with labels from the
train ground truth only. It is kept conservative because macro F0.5 punishes
false merges more than missed matches:
  * shallow trees, strong regularisation, row/column subsampling;
  * monotone constraints (a higher name/address similarity can never lower the
    match probability), which also helps it transfer to the unseen country;
  * a probability threshold tuned for macro F0.5 on held-out S1 entities;
  * one-to-one clean-up: each S2/S3 record goes to at most one S1 entity (a
    property of the ground truth), keeping the most probable claim.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

from src.config import BETA, SEED
from src.evaluate import f_beta
from src.features import FEATURE_COLUMNS, build_features

# +1: probability may only rise with the feature, -1: only fall, 0: free.
_MONOTONE = {
    "name_ratio": 1, "name_token_set": 1, "name_token_sort": 1, "name_partial": 1,
    "name_jaro_winkler": 1, "name_core_jaccard": 1, "name_tfidf": 1,
    "addr_token_set": 1, "addr_ratio": 1, "addr_tfidf": 1,
    "house_num": 1, "name_missing": 0, "addr_missing": 0,
    "country_match": 1, "block_rank": -1, "block_score": 1,
}

PARAMS = dict(
    objective="binary",
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=5.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)

DEFAULT_THRESHOLD = 0.5  # replaced by the tuned value stored in the bundle


# ---------------------------------------------------------------- model -----
def train(features: pd.DataFrame, labels: pd.Series, eval_set=None,
          params: dict | None = None) -> LGBMClassifier:
    """Fit the pair classifier. `eval_set=(X_val, y_val)` enables early
    stopping on held-out pairs."""
    p = {**PARAMS, **(params or {})}
    p["monotone_constraints"] = [_MONOTONE.get(c, 0) for c in features.columns]
    model = LGBMClassifier(**p)
    kw = {}
    if eval_set is not None:
        kw = dict(eval_set=[eval_set], eval_metric="binary_logloss",
                  callbacks=[early_stopping(50, verbose=False)])
    model.fit(features, labels, **kw)
    return model


def predict_proba(model, features: pd.DataFrame) -> np.ndarray:
    """Match probability per pair, same order as `features`."""
    return model.predict_proba(features[FEATURE_COLUMNS])[:, 1]


def select_matches(pairs: pd.DataFrame, proba, threshold: float,
                   one_to_one: bool = True) -> dict:
    """Keep pairs with proba >= threshold -> {s1_id: [match ids]}.
    With `one_to_one`, a candidate id is kept only for its most probable S1."""
    df = pd.DataFrame({"s1_id": pairs["s1_id"].to_numpy(),
                       "cand_id": pairs["cand_id"].to_numpy(),
                       "p": np.asarray(proba, dtype=float)})
    df = df[df["p"] >= threshold]
    if one_to_one:
        df = df.sort_values("p", ascending=False, kind="stable").drop_duplicates("cand_id")
    out: dict[str, list] = {}
    for s1, cand in zip(df["s1_id"], df["cand_id"]):
        out.setdefault(s1, []).append(cand)
    return out


# ---------------------------------------------------------------- evaluation
def evaluate_matches(pred: dict, truth: dict, beta: float = BETA) -> dict:
    """Scores for one set of predictions against {s1_id: set(match ids)}.

    macro_f05 is the competition metric (per S1 entity, singletons included).
    precision/recall are pair-level over all S1 entities in `truth`.
    macro_precision is averaged over entities with >=1 prediction;
    macro_recall over entities with >=1 true match; singleton_acc is the share
    of true singletons left empty.
    """
    tp = n_pred = n_true = 0
    f_sum = p_sum = r_sum = 0.0
    p_n = r_n = single_n = single_ok = 0
    for s1, t in truth.items():
        pr = set(pred.get(s1, ()))
        t = set(t)
        hit = len(pr & t)
        tp, n_pred, n_true = tp + hit, n_pred + len(pr), n_true + len(t)
        f_sum += f_beta(pr, t, beta)
        if pr:
            p_sum += hit / len(pr)
            p_n += 1
        if t:
            r_sum += hit / len(t)
            r_n += 1
        else:
            single_n += 1
            single_ok += not pr
    return {
        "precision": tp / n_pred if n_pred else 1.0,
        "recall": tp / n_true if n_true else 1.0,
        "macro_precision": p_sum / p_n if p_n else 1.0,
        "macro_recall": r_sum / r_n if r_n else 1.0,
        "singleton_acc": single_ok / single_n if single_n else 1.0,
        "macro_f05": f_sum / len(truth) if truth else 0.0,
        "pred_pairs": n_pred,
    }


def threshold_report(pairs: pd.DataFrame, proba, truth: dict,
                     thresholds=None, one_to_one: bool = True) -> pd.DataFrame:
    """threshold -> precision -> recall -> macro F0.5 table on validation."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.96, 0.05), 2)
    rows = []
    for t in thresholds:
        pred = select_matches(pairs, proba, float(t), one_to_one=one_to_one)
        rows.append({"threshold": float(t), **evaluate_matches(pred, truth)})
    return pd.DataFrame(rows)


def best_threshold(report: pd.DataFrame) -> float:
    """Threshold with the highest macro F0.5 (ties -> the higher threshold)."""
    top = report["macro_f05"].max()
    return float(report.loc[report["macro_f05"] >= top - 1e-12, "threshold"].max())


# ---------------------------------------------------------------- bundle ----
def save_bundle(path: Path, model, vectorizers: dict, threshold: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "vectorizers": vectorizers, "threshold": threshold,
                 "features": FEATURE_COLUMNS}, path)


def load_bundle(path: Path) -> dict:
    return joblib.load(path)


def predict_matches(bundle: dict, pairs: pd.DataFrame, s1: pd.DataFrame,
                    others: pd.DataFrame, chunk_size: int = 500_000,
                    threshold: float | None = None) -> dict:
    """End-to-end scoring for integration (predict.py): features in chunks to
    bound memory, then threshold + one-to-one selection over all pairs."""
    # Keep each S1's candidates in one chunk so within-S1 ranks are correct.
    pairs = pairs.sort_values("s1_id", kind="stable").reset_index(drop=True)
    proba = np.empty(len(pairs), dtype=np.float32)
    ids = pairs["s1_id"].to_numpy()
    start = 0
    while start < len(pairs):
        end = min(start + chunk_size, len(pairs))
        while end < len(pairs) and ids[end] == ids[end - 1]:
            end += 1
        chunk = pairs.iloc[start:end]
        feats = build_features(chunk, s1, others, bundle["vectorizers"])
        proba[start:end] = predict_proba(bundle["model"], feats)
        start = end
    t = bundle["threshold"] if threshold is None else threshold
    return select_matches(pairs, proba, t)
