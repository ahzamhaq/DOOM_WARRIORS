"""Candidate generation. Owner: Person 2.

Contract: returns a DataFrame of candidate pairs with columns [s1_id, cand_id].
This exact set is what the matcher scores AND what goes into candidate_pairs.tsv.

Plan (from EDA):
  - Hard block on `country` (100% of true pairs share it). Loop country by country
    to keep memory low; never hard-code the list of countries.
  - Union of two retrievers within each country, top-K each:
      1. name: TF-IDF char n-grams on name_norm, sparse top-K cosine (chunked)
      2. address: TF-IDF on addr_norm (catches Indic-script / domain-style names)
  - Tune K on train: report candidate recall + avg candidates per S1.
"""
import pandas as pd


def generate_candidates(s1: pd.DataFrame, others: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """s1: normalized Source 1 records; others: normalized Source 2 + 3 records."""
    raise NotImplementedError
