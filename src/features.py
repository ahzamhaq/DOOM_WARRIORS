"""Pairwise similarity features. Owner: Person 3.  Contract: docs/CONTRACT.md §5.

Country-agnostic (test adds France). No labels, no `country` string, no ID digits as features.
Starting set (rapidfuzz): name token_set_ratio / ratio / jaro-winkler, name token jaccard, address
token_set_ratio, house-number match, city/state token overlap, address-empty flags, name-is-domain flag,
script-mismatch flag, source (S2/S3 from ID prefix), plus config.BLK_COLUMNS passed through.
"""
import pandas as pd

FEATURE_NAMES: list[str] = []  # owner fills in; fixed order, exported for matcher/evaluate


def build_features(pairs: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """`pairs`: s1_id, cand_id, blk_*. `s1`, `pool`: normalized records indexed by entity_id.

    Returns float32 features, same index and order as `pairs`, columns == FEATURE_NAMES.
    """
    raise NotImplementedError
