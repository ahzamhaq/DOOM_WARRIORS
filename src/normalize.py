"""Name / address normalization. Owner: Person 2.  Contract: docs/CONTRACT.md §3.

Row-wise, stateless, deterministic; never branch on the country value (France is unseen in train).
EDA findings to handle (see outputs/eda_report.txt):
  - junk prefixes/wrappers: "-- ", "<< ", "#", "[...]"
  - names given as domains/handles: "aimsons.com", "@barretocardiology"
  - legal suffix variants: Pvt/Private, Ltd/Limited, Inc, LLC/L.L.C., PLLC, SARL, SAS ...
  - "&" vs "and", case differences (S2 addresses are UPPERCASE)
  - Indic-script names (Devanagari, Telugu, Bengali, Tamil, Malayalam) in S2/S3 India
  - address: Rd/Road, St/Street, full state name vs code, "NULL" tokens, component reordering
"""
import pandas as pd


def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with config.NORM_COLUMNS added; same index and row order, raw columns untouched."""
    raise NotImplementedError
