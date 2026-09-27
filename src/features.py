"""Pairwise similarity features. Owner: Person 3.

Contract: takes candidate pairs [s1_id, cand_id] (optionally with `rank` and/or
`score` columns from blocking) plus the Source 1 and Source 2+3 records, and
returns one numeric feature row per pair, in the same order as `pairs`.

All features are country-agnostic: the only country feature is whether the two
labels agree, so nothing is tied to US/India and France (test-only) is handled
the same way.

Works with or without normalize.py: if the records already carry `name_norm` /
`addr_norm` (from normalize.add_normalized_columns) those are used, otherwise a
small internal normalizer is applied. The feature set is identical either way.

Features (16):
  name   : name_ratio, name_token_set, name_token_sort, name_partial,
           name_jaro_winkler, name_core_jaccard, name_tfidf
  address: addr_token_set, addr_ratio, addr_tfidf
  house  : house_num (1 = same number, 0 = unknown, -1 = both present but differ)
  empty  : name_missing, addr_missing (number of sides with an empty field, 0-2)
  country: country_match
  search : block_rank (0 = best candidate for that S1), block_score
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer

try:  # ISC-licensed transliterator: Devanagari/Tamil/... -> Latin letters
    from anyascii import anyascii as _translit
except ImportError:  # fall back to accent stripping only
    def _translit(s: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFKD", s)
                       if not unicodedata.combining(c))

FEATURE_COLUMNS = [
    "name_ratio", "name_token_set", "name_token_sort", "name_partial",
    "name_jaro_winkler", "name_core_jaccard", "name_tfidf",
    "addr_token_set", "addr_ratio", "addr_tfidf",
    "house_num",
    "name_missing", "addr_missing",
    "country_match",
    "block_rank", "block_score",
]

# Generic legal-form / filler words, dropped for the "core name" comparison.
# Includes French forms so the feature behaves sensibly on the unseen country.
_LEGAL = {
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "corp",
    "corporation", "co", "company", "pvt", "private", "plc", "pllc", "pc",
    "the", "and", "of", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc",
    "groupe", "group", "gmbh", "dba", "et", "cie", "com", "www",
}
_ADDR_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane",
    "ct": "court", "hwy": "highway", "pkwy": "parkway", "pl": "place",
    "sq": "square", "ste": "suite", "apt": "apartment", "fl": "floor",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "nr": "near", "opp": "opposite", "sec": "sector", "ph": "phase",
    "r": "rue", "ch": "chemin", "rte": "route", "null": "",
}
_NAME_ABBR = {
    "pvt": "private", "pte": "private", "ltd": "limited", "ltda": "limited",
    "corp": "corporation", "inc": "incorporated", "co": "company",
    "intl": "international", "mfg": "manufacturing", "svcs": "services",
    "svc": "service", "mgmt": "management", "assoc": "associates",
    "bros": "brothers", "ent": "enterprises",
}
_NON_WORD = re.compile(r"[^0-9a-z]+")
_HOUSE_NUM = re.compile(r"\d+")


def _norm_text(s: str) -> str:
    s = _translit(s or "").lower().replace("&", " and ")
    return _NON_WORD.sub(" ", s).strip()


def normalize_name(values) -> list[str]:
    out = []
    for v in values:
        t = _norm_text(v)
        t = re.sub(r"\bl l c\b", "llc", t)
        t = re.sub(r"\bl l p\b", "llp", t)
        toks = [_NAME_ABBR.get(w, w) for w in t.split()]
        out.append(" ".join(w for w in toks if w))
    return out


def normalize_address(values) -> list[str]:
    out = []
    for v in values:
        toks = [_ADDR_ABBR.get(t, t) for t in _norm_text(v).split()]
        out.append(" ".join(t for t in toks if t))
    return out


def _house_number(addr: str) -> str:
    """First number in the address (street / building number). Postcodes are
    usually at the end, so the first number avoids most of them."""
    m = _HOUSE_NUM.search(addr or "")
    return m.group(0).lstrip("0") if m else ""


# ---------------------------------------------------------------- TF-IDF ----
def fit_vectorizers(s1: pd.DataFrame, others: pd.DataFrame,
                    max_docs: int = 400_000, seed: int = 42) -> dict:
    """Fit the character n-gram TF-IDF vectorizers once, on training text, so
    train, validation and test features all use the same IDF weights."""
    rng = np.random.default_rng(seed)
    names, addrs = [], []
    for df in (s1, others):
        n = len(df)
        take = np.sort(rng.choice(n, size=min(n, max_docs // 2), replace=False))
        sub = df.iloc[take]
        names += _get_norm(sub, "name")
        addrs += _get_norm(sub, "addr")
    kw = dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
              max_features=300_000, dtype=np.float32)
    return {
        "name": TfidfVectorizer(**kw).fit(names),
        "addr": TfidfVectorizer(**kw).fit([a for a in addrs if a] or ["x"]),
    }


def _tfidf_cos(vec, a: list[str], b: list[str]) -> np.ndarray:
    xa, xb = vec.transform(a), vec.transform(b)  # rows are L2-normalised
    return np.asarray(xa.multiply(xb).sum(axis=1), dtype=np.float32).ravel()


# ---------------------------------------------------------------- helpers ---
def _get_norm(df: pd.DataFrame, kind: str) -> list[str]:
    col, raw = ("name_norm", "business_name") if kind == "name" else ("addr_norm", "business_address")
    if col in df.columns:
        return df[col].fillna("").astype(str).tolist()
    vals = df[raw].fillna("").astype(str).tolist()
    return normalize_name(vals) if kind == "name" else normalize_address(vals)


def _side(ids: pd.Series, records: pd.DataFrame) -> pd.DataFrame:
    """Records aligned to `ids` (one row per pair). Each distinct record is
    normalized once, however many pairs it appears in."""
    uniq = pd.unique(ids.to_numpy())
    rec = records[records["entity_id"].isin(uniq)].drop_duplicates("entity_id")
    small = pd.DataFrame({
        "name": _get_norm(rec, "name"),
        "addr": _get_norm(rec, "addr"),
        "house": [_house_number(x) for x in rec["business_address"].fillna("").astype(str)],
        "country": rec["country"].fillna("").astype(str).str.strip().str.lower().to_numpy(),
    }, index=rec["entity_id"].to_numpy())
    return small.reindex(ids.to_numpy()).fillna("")


def _core_jaccard(a: str, b: str) -> float:
    ta = {t for t in a.split() if t not in _LEGAL}
    tb = {t for t in b.split() if t not in _LEGAL}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _house_state(a: str, b: str) -> int:
    if not a or not b:
        return 0
    return 1 if a == b else -1


# ---------------------------------------------------------------- main ------
def build_features(pairs: pd.DataFrame, s1: pd.DataFrame, others: pd.DataFrame,
                   vectorizers: dict | None = None) -> pd.DataFrame:
    """One feature row per pair (same order as `pairs`).

    `vectorizers` should be the dict from fit_vectorizers() on training data
    (matcher.train stores it in the model bundle). If omitted, vectorizers are
    fitted on the text of these pairs; fine for a quick look, but TF-IDF
    weights then differ between batches.
    """
    pairs = pairs.reset_index(drop=True)
    a = _side(pairs["s1_id"], s1)
    b = _side(pairs["cand_id"], others)
    an, bn = a["name"].tolist(), b["name"].tolist()
    aa, ba = a["addr"].tolist(), b["addr"].tolist()

    if vectorizers is None:
        kw = dict(analyzer="char_wb", ngram_range=(2, 4), dtype=np.float32)
        vectorizers = {"name": TfidfVectorizer(**kw).fit(an + bn + ["x"]),
                       "addr": TfidfVectorizer(**kw).fit(aa + ba + ["x"])}

    f = {}
    f["name_ratio"] = [fuzz.ratio(x, y) for x, y in zip(an, bn)]
    f["name_token_set"] = [fuzz.token_set_ratio(x, y) for x, y in zip(an, bn)]
    f["name_token_sort"] = [fuzz.token_sort_ratio(x, y) for x, y in zip(an, bn)]
    f["name_partial"] = [fuzz.partial_ratio(x, y) for x, y in zip(an, bn)]
    f["name_jaro_winkler"] = [JaroWinkler.normalized_similarity(x, y) for x, y in zip(an, bn)]
    f["name_core_jaccard"] = [_core_jaccard(x, y) for x, y in zip(an, bn)]
    f["name_tfidf"] = _tfidf_cos(vectorizers["name"], an, bn)

    f["addr_token_set"] = [fuzz.token_set_ratio(x, y) if x and y else 0.0 for x, y in zip(aa, ba)]
    f["addr_ratio"] = [fuzz.ratio(x, y) if x and y else 0.0 for x, y in zip(aa, ba)]
    f["addr_tfidf"] = _tfidf_cos(vectorizers["addr"], aa, ba)

    f["house_num"] = [_house_state(x, y) for x, y in zip(a["house"], b["house"])]

    f["name_missing"] = (a["name"].eq("").to_numpy(int) + b["name"].eq("").to_numpy(int))
    f["addr_missing"] = (a["addr"].eq("").to_numpy(int) + b["addr"].eq("").to_numpy(int))

    f["country_match"] = (a["country"].to_numpy() == b["country"].to_numpy()).astype(int)

    # Blocking / search signals. Blocking's contract only guarantees
    # [s1_id, cand_id]; if it also passes `rank` and/or `score` they are used.
    # Otherwise the candidate's rank by name TF-IDF within its S1 group stands
    # in as the search rank.
    if "score" in pairs.columns:
        score = pairs["score"].astype(float).to_numpy()
    else:
        score = np.asarray(f["name_tfidf"], dtype=float)
    if "rank" in pairs.columns:
        rank = pairs["rank"].astype(float).to_numpy()
    else:
        rank = (pd.Series(score).groupby(pairs["s1_id"].to_numpy())
                .rank(method="first", ascending=False).to_numpy() - 1)
    f["block_rank"] = rank
    f["block_score"] = score

    out = pd.DataFrame(f, columns=FEATURE_COLUMNS).astype(np.float32)
    for c in ("name_ratio", "name_token_set", "name_token_sort", "name_partial",
              "addr_token_set", "addr_ratio"):
        out[c] /= 100.0
    return out
