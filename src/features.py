"""Pairwise similarity features. Owner: Person 3.  Contract: docs/CONTRACT.md §5.

build_features(pairs, s1, pool) -> feats: one float32 row per pair, same index and order as
`pairs`, columns == FEATURE_NAMES. Depends only on the pair, the two records and corpus
statistics computed from `pool`. No labels, no country string, no ID digits.

Uses the normalize.py columns (config.NORM_COLUMNS) when present. While normalize.py is
still a stub, the same columns are derived here, so the module runs either way.
Names/addresses are also transliterated to Latin letters (anyascii, ISC licence), so a
Devanagari name and its English spelling can be compared.

Features (21):
  name    : name_ratio, name_token_set, name_token_sort, name_partial,
            name_jaro_winkler, name_core_jaccard, name_tfidf
  address : addr_token_set, addr_ratio, addr_tfidf
  house   : house_num      1 same number, 0 unknown, -1 both present but different
  empty   : name_missing, addr_missing   (sides with an empty field, 0-2)
  country : country_match  (1 when both labels agree; the label itself is never used)
  other   : script_mismatch, domain_name, cand_is_s3
  blocking: blk_name_score, blk_addr_score, blk_name_rank, blk_addr_rank (passed through)
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import BLK_COLUMNS, CAND_ID, S1_ID

try:  # ISC-licensed transliterator: Devanagari/Tamil/... -> Latin letters
    from anyascii import anyascii as _translit
except ImportError:  # accent stripping only
    def _translit(s: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

FEATURE_NAMES: list[str] = [
    "name_ratio", "name_token_set", "name_token_sort", "name_partial",
    "name_jaro_winkler", "name_core_jaccard", "name_tfidf",
    "addr_token_set", "addr_ratio", "addr_tfidf",
    "house_num", "name_missing", "addr_missing", "country_match",
    "script_mismatch", "domain_name", "cand_is_s3",
    *BLK_COLUMNS,
]

# Legal-form / filler words ignored by name_core_jaccard (French forms included for the unseen country).
_LEGAL = {
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "corp", "corporation", "co",
    "company", "pvt", "private", "plc", "pllc", "pc", "the", "and", "of", "sarl", "sas", "sasu",
    "sa", "eurl", "sci", "snc", "groupe", "group", "gmbh", "dba", "et", "cie", "com", "www",
}
_NAME_ABBR = {
    "pvt": "private", "pte": "private", "ltd": "limited", "ltda": "limited", "corp": "corporation",
    "inc": "incorporated", "co": "company", "intl": "international", "mfg": "manufacturing",
    "svcs": "services", "svc": "service", "mgmt": "management", "assoc": "associates",
    "bros": "brothers", "ent": "enterprises",
}
_ADDR_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "hwy": "highway", "pkwy": "parkway", "pl": "place", "sq": "square", "ste": "suite",
    "apt": "apartment", "fl": "floor", "n": "north", "s": "south", "e": "east", "w": "west",
    "nr": "near", "opp": "opposite", "sec": "sector", "ph": "phase", "r": "rue",
    "ch": "chemin", "rte": "route", "null": "",
}
_NON_WORD = re.compile(r"[^0-9a-z]+")
_FIRST_NUM = re.compile(r"\d+")
_DOMAIN = re.compile(r"(^@)|(\.(com|net|org|in|co|fr|biz|io|us|info)\b)|(\bwww\b)", re.I)
_INDIC = re.compile(r"[ऀ-෿]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")

_TFIDF_DOCS = 150_000  # pool rows used for IDF statistics (deterministic sample)


# ------------------------------------------------------------- normalization
def _canon(s: str, abbr: dict) -> str:
    s = _translit(s or "").lower().replace("&", " and ")
    s = _NON_WORD.sub(" ", s)
    s = re.sub(r"\bl l ([cp])\b", r"ll\1", s)
    return " ".join(w for w in (abbr.get(t, t) for t in s.split()) if w)


def canon_name(values) -> list[str]:
    return [_canon(v, _NAME_ABBR) for v in values]


def canon_addr(values) -> list[str]:
    return [_canon(v, _ADDR_ABBR) for v in values]


def _script(s: str) -> str:
    if _INDIC.search(s or ""):
        return "indic"
    return "latin" if _LATIN.search(s or "") or not s else "other"


def _side(ids: np.ndarray, rec: pd.DataFrame) -> pd.DataFrame:
    """Per-pair view of one side. Each distinct record is processed once."""
    uniq = pd.unique(ids)
    r = rec.loc[rec.index.intersection(uniq)]
    r = r[~r.index.duplicated()]
    name_src = r["name_norm"] if "name_norm" in r.columns else r["business_name"]
    addr_src = r["addr_norm"] if "addr_norm" in r.columns else r["business_address"]
    raw_name = r["business_name"].astype(str)
    raw_addr = r["business_address"].astype(str)
    if "house_no" in r.columns:
        house = r["house_no"].astype(str).str.lstrip("0").tolist()
    else:
        house = [(m.group(0).lstrip("0") if (m := _FIRST_NUM.search(a)) else "") for a in raw_addr]
    small = pd.DataFrame({
        "name": canon_name(name_src.astype(str)),
        "addr": canon_addr(addr_src.astype(str)),
        "house": house,
        "script": (r["name_script"].astype(str).tolist() if "name_script" in r.columns
                   else [_script(x) for x in raw_name]),
        "domain": (r["name_is_domain"].astype(bool).tolist() if "name_is_domain" in r.columns
                   else [bool(_DOMAIN.search(x)) for x in raw_name]),
        "country": r["country"].astype(str).str.strip().str.lower().tolist(),
    }, index=r.index)
    return small.reindex(ids)


# ------------------------------------------------------------- similarities
def _pool_vectorizers(pool: pd.DataFrame) -> dict:
    """Char n-gram TF-IDF fitted on a deterministic sample of the pool (static corpus stats:
    the same for every S1 chunk of a country)."""
    idx = pool.index.sort_values()
    step = max(1, len(idx) // _TFIDF_DOCS)
    sample = pool.loc[idx[::step]]
    names = canon_name((sample["name_norm"] if "name_norm" in sample else sample["business_name"]).astype(str))
    addrs = canon_addr((sample["addr_norm"] if "addr_norm" in sample else sample["business_address"]).astype(str))
    kw = dict(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=200_000, dtype=np.float32)
    out = {}
    for key, docs in (("name", names), ("addr", [a for a in addrs if a])):
        try:
            out[key] = TfidfVectorizer(**kw).fit(docs or ["x"])
        except ValueError:  # tiny pools (e.g. toy tests): no n-gram reaches min_df
            out[key] = TfidfVectorizer(**{**kw, "min_df": 1}).fit(docs or ["x"])
    return out


def _tfidf_cos(vec, a: list[str], b: list[str]) -> np.ndarray:
    xa, xb = vec.transform(a), vec.transform(b)  # rows are L2-normalised
    return np.asarray(xa.multiply(xb).sum(axis=1), dtype=np.float32).ravel()


def _core_jaccard(a: str, b: str) -> float:
    ta = {t for t in a.split() if t not in _LEGAL}
    tb = {t for t in b.split() if t not in _LEGAL}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _pairwise(fn, a, b, skip_empty=False) -> np.ndarray:
    if skip_empty:
        return np.fromiter((fn(x, y) if x and y else 0.0 for x, y in zip(a, b)), np.float32, len(a))
    return np.fromiter((fn(x, y) for x, y in zip(a, b)), np.float32, len(a))


# ------------------------------------------------------------- main
def build_features(pairs: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """`pairs`: s1_id, cand_id, blk_*. `s1`, `pool`: normalized records indexed by entity_id.

    Returns float32 features, same index and order as `pairs`, columns == FEATURE_NAMES.
    """
    n = len(pairs)
    if n == 0:
        return pd.DataFrame({c: pd.Series(dtype=np.float32) for c in FEATURE_NAMES}, index=pairs.index)
    a = _side(pairs[S1_ID].to_numpy(), s1)
    b = _side(pairs[CAND_ID].to_numpy(), pool)
    an, bn = a["name"].fillna("").tolist(), b["name"].fillna("").tolist()
    aa, ba = a["addr"].fillna("").tolist(), b["addr"].fillna("").tolist()
    vec = _pool_vectorizers(pool)

    f: dict[str, np.ndarray] = {}
    f["name_ratio"] = _pairwise(fuzz.ratio, an, bn) / 100
    f["name_token_set"] = _pairwise(fuzz.token_set_ratio, an, bn) / 100
    f["name_token_sort"] = _pairwise(fuzz.token_sort_ratio, an, bn) / 100
    f["name_partial"] = _pairwise(fuzz.partial_ratio, an, bn) / 100
    f["name_jaro_winkler"] = _pairwise(JaroWinkler.normalized_similarity, an, bn)
    f["name_core_jaccard"] = _pairwise(_core_jaccard, an, bn)
    f["name_tfidf"] = _tfidf_cos(vec["name"], an, bn)

    f["addr_token_set"] = _pairwise(fuzz.token_set_ratio, aa, ba, skip_empty=True) / 100
    f["addr_ratio"] = _pairwise(fuzz.ratio, aa, ba, skip_empty=True) / 100
    f["addr_tfidf"] = _tfidf_cos(vec["addr"], aa, ba)

    ha, hb = a["house"].fillna("").to_numpy(), b["house"].fillna("").to_numpy()
    both = (ha != "") & (hb != "")
    f["house_num"] = np.where(both, np.where(ha == hb, 1.0, -1.0), 0.0)
    f["name_missing"] = (np.array(an) == "").astype(np.float32) + (np.array(bn) == "")
    f["addr_missing"] = (np.array(aa) == "").astype(np.float32) + (np.array(ba) == "")
    f["country_match"] = (a["country"].to_numpy() == b["country"].to_numpy()).astype(np.float32)
    f["script_mismatch"] = (a["script"].to_numpy() != b["script"].to_numpy()).astype(np.float32)
    f["domain_name"] = (a["domain"].fillna(False).to_numpy(bool) | b["domain"].fillna(False).to_numpy(bool)
                        ).astype(np.float32)
    f["cand_is_s3"] = pairs[CAND_ID].astype(str).str.startswith("S3-").to_numpy(np.float32)
    for c in BLK_COLUMNS:  # blocking's retrieval signals, passed through (NaN if absent)
        f[c] = (pd.to_numeric(pairs[c], errors="coerce").to_numpy(np.float32) if c in pairs.columns
                else np.full(n, np.nan, np.float32))

    out = pd.DataFrame(f, index=pairs.index)[FEATURE_NAMES].astype(np.float32)
    return out.replace([np.inf, -np.inf], np.nan)
