"""Pair classifier + threshold -> final matches. Owner: Person 3.  Contract: docs/CONTRACT.md §6.

Plan: LightGBM binary classifier on features.py output, labels from evaluate.label_pairs. The threshold
is tuned for macro F0.5 (evaluate.sweep_thresholds), not pair-level accuracy, then stored on the model.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ONE_TO_ONE


class Model:
    feature_names: list[str]
    threshold: float = 0.5

    def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
        """P(match) per row, float32, shape (n,). Must reorder columns by `feature_names`."""
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "Model":
        raise NotImplementedError


def train(feats: pd.DataFrame, labels: pd.Series) -> Model:
    """`labels`: int8 1/0 aligned to `feats.index`."""
    raise NotImplementedError


def select_matches(scored: pd.DataFrame, threshold: float, one_to_one: bool = ONE_TO_ONE) -> pd.DataFrame:
    """`scored`: s1_id, cand_id, proba for ONE whole country. Returns matches: s1_id, cand_id.

    Keep proba >= threshold. If `one_to_one` (EXPERIMENTAL, off by default): a cand_id may then appear for
    only one S1 (keep the highest proba; ties -> smallest s1_id). Compare validation F0.5 with and without.
    """
    raise NotImplementedError
