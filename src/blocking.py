"""Candidate generation. Owner: Person 2.  Contract: docs/CONTRACT.md §4.

Plan (from EDA):
  - Everything here is already within ONE country (predict.py loops over countries).
  - Union of two retrievers, top-K each:
      1. name: TF-IDF char n-grams on name_norm, sparse top-K cosine
      2. address: same on addr_norm (catches Indic-script / domain-style names)
  - Tune K on train: report candidate recall + avg candidates per S1 (evaluate.blocking_report).
  - Same code and parameters at train and test time, so blk_* features have the same distribution.
"""
import pandas as pd

from src.config import BLOCK_K, MAX_CANDS_PER_S1


class BlockIndex:
    """Whatever the retrievers need (vectorizers, sparse matrices). Built once per country."""


def build_index(pool: pd.DataFrame) -> BlockIndex:
    """`pool`: normalized S2+S3 records of one country, indexed by entity_id."""
    raise NotImplementedError


def generate_candidates(
    s1: pd.DataFrame, index: BlockIndex, k: int = BLOCK_K, max_cands: int = MAX_CANDS_PER_S1
) -> pd.DataFrame:
    """`s1`: normalized S1 chunk. Returns pairs: s1_id, cand_id + config.BLK_COLUMNS.

    Same country only, S2-/S3- ids only, no duplicate pairs, at most `max_cands` per S1.
    S1 entities with no candidates just have no rows.
    """
    raise NotImplementedError
