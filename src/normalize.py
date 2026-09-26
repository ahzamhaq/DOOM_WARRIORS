"""Name / address normalization. Owner: Person 2.

Must work for any country label (test adds France, unseen in train).
EDA findings to handle (see outputs/eda_report.txt):
  - junk prefixes/wrappers: "-- ", "<< ", "#", "[...]"
  - names given as domains/handles: "aimsons.com", "@barretocardiology"
  - legal suffix variants: Pvt/Private, Ltd/Limited, Inc, LLC/L.L.C., PLLC, SARL, SAS ...
  - "&" vs "and", case differences (S2 addresses are UPPERCASE)
  - Indic-script names (Devanagari, Telugu, Bengali, Tamil, Malayalam) in S2/S3 India
  - address: Rd/Road, St/Street, full state name vs code, "NULL" tokens, component reordering
"""
import pandas as pd


def normalize_name(s: pd.Series) -> pd.Series:
    """Lowercase, strip junk/punctuation, drop legal suffixes. Returns cleaned string."""
    raise NotImplementedError


def normalize_address(s: pd.Series) -> pd.Series:
    """Lowercase, expand/standardize abbreviations, drop punctuation and NULL tokens."""
    raise NotImplementedError


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `name_norm` and `addr_norm` columns to a source DataFrame."""
    df["name_norm"] = normalize_name(df["business_name"])
    df["addr_norm"] = normalize_address(df["business_address"])
    return df
