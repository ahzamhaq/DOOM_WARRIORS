"""Pairwise similarity features. Owner: Person 3.

Contract: takes candidate pairs [s1_id, cand_id] plus the normalized records and
returns one numeric feature row per pair (same order). Keep features
country-agnostic so they transfer to France.

Starting set (rapidfuzz): name token_set_ratio / ratio / jaro-winkler, name token
jaccard, address token_set_ratio, house-number match, city/state token overlap,
address-empty flags, name-is-domain flag, script-mismatch flag, candidate source
(S2/S3), retrieval rank/score from blocking.
"""
import pandas as pd


def build_features(pairs: pd.DataFrame, s1: pd.DataFrame, others: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError
