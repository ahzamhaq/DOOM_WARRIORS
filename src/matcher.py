"""Pair classifier + threshold -> final matches. Owner: Person 3.  Contract: docs/CONTRACT.md §6.

LightGBM binary classifier on features.py output (labels from the train ground truth only),
kept conservative because macro F0.5 punishes false merges more than missed matches:
  * shallow trees, strong L1/L2 regularisation, row/column subsampling;
  * monotone constraints: a higher name/address similarity can never lower P(match), which also
    keeps behaviour sensible on the unseen country;
  * a probability threshold tuned for macro F0.5 on held-out S1 entities, stored on the Model.
select_matches supports the EXPERIMENTAL one-to-one rule (off by default, config.ONE_TO_ONE).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

from src.config import CAND_ID, ONE_TO_ONE, PROBA, S1_ID, SEED

# +1: P(match) may only rise with the feature; -1: only fall; 0: unconstrained.
MONOTONE = {
    "name_ratio": 1, "name_token_set": 1, "name_token_sort": 1, "name_partial": 1,
    "name_jaro_winkler": 1, "name_core_jaccard": 1, "name_tfidf": 1,
    "addr_token_set": 1, "addr_ratio": 1, "addr_tfidf": 1, "house_num": 1,
    "country_match": 1, "blk_name_score": 1, "blk_addr_score": 1,
    "addr_num_jaccard": 1, "addr_num_conflict": -1,
}

PARAMS = dict(
    objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31, max_depth=6,
    min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1,
)


class Model:
    """Trained classifier + the feature order it expects + the tuned decision threshold."""

    def __init__(self, clf: LGBMClassifier, feature_names: list[str], threshold: float = 0.5):
        self.clf = clf
        self.feature_names = list(feature_names)
        self.threshold = float(threshold)

    def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
        """P(match) per row, float32, shape (n,). Columns are reordered by `feature_names`."""
        if len(feats) == 0:
            return np.zeros(0, dtype=np.float32)
        x = feats[self.feature_names].astype(np.float32)
        return self.clf.predict_proba(x)[:, 1].astype(np.float32)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "feature_names": self.feature_names,
                     "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path: Path) -> "Model":
        d = joblib.load(path)
        return cls(d["clf"], d["feature_names"], d["threshold"])


def train(feats: pd.DataFrame, labels: pd.Series, eval_feats: pd.DataFrame | None = None,
          eval_labels: pd.Series | None = None, params: dict | None = None) -> Model:
    """`labels`: int8 1/0 aligned to `feats.index`. Optional eval set (other S1 entities than the
    training ones) enables early stopping. The threshold is set afterwards by the caller."""
    p = {**PARAMS, **(params or {})}
    cols = list(feats.columns)
    p["monotone_constraints"] = [MONOTONE.get(c, 0) for c in cols]
    clf = LGBMClassifier(**p)
    y = labels.loc[feats.index].astype(np.int8)
    kw = {}
    if eval_feats is not None and len(eval_feats):
        kw = dict(eval_set=[(eval_feats[cols], eval_labels.loc[eval_feats.index].astype(np.int8))],
                  eval_metric="binary_logloss", callbacks=[early_stopping(50, verbose=False)])
    clf.fit(feats, y, **kw)
    return Model(clf, cols)


def select_matches(scored: pd.DataFrame, threshold: float, one_to_one: bool = ONE_TO_ONE) -> pd.DataFrame:
    """`scored`: s1_id, cand_id, proba for ONE whole country. Returns matches: s1_id, cand_id.

    Keep proba >= threshold. If `one_to_one` (EXPERIMENTAL, off by default): a cand_id may then
    appear for only one S1 (keep the highest proba; ties -> smallest s1_id).
    """
    kept = scored.loc[scored[PROBA] >= threshold, [S1_ID, CAND_ID, PROBA]]
    if one_to_one and len(kept):
        kept = (kept.sort_values([PROBA, S1_ID], ascending=[False, True], kind="stable")
                    .drop_duplicates(CAND_ID, keep="first"))
    return kept[[S1_ID, CAND_ID]].reset_index(drop=True)
