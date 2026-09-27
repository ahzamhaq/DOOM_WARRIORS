"""Pairwise similarity features. Owner: Person 3.  Contract: docs/CONTRACT.md §5.

build_features(pairs, s1, pool) -> feats: one float32 row per pair, same index and order as
`pairs`, columns == FEATURE_NAMES. Depends only on the pair, the two records and corpus
statistics computed from `pool`. No labels, no country string, no ID digits.

Uses the normalize.py columns (config.NORM_COLUMNS) when present; while normalize.py is still a
stub the same information is derived here. Names/addresses are transliterated to Latin letters
with anyascii (ISC licence, REQUIRED: without it Indic-script names become empty strings), so a
Devanagari name and its English spelling can be compared.

Performance: every distinct record is canonicalized and TF-IDF-transformed once (not once per
pair); string similarities run in rapidfuzz.process.cpdist (C++, all cores); TF-IDF cosine is a
sparse row-wise product over blocks of pairs, so memory stays bounded.

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

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

from src.config import BLK_COLUMNS, CAND_ID, S1_ID

try:
    from anyascii import anyascii as _translit
except ImportError as exc:  # no silent fallback: it would blank every Indic-script name
    raise ImportError(
        "src/features.py requires the 'anyascii' package (tested with anyascii==0.3.3; "
        "pip install -r requirements.txt). Without it Devanagari/Tamil/... names become empty "
        "strings and every name feature is zero for those pairs."
    ) from exc

FEATURE_NAMES: list[str] = [
    "name_ratio", "name_token_set", "name_token_sort", "name_partial",
    "name_jaro_winkler", "name_core_jaccard", "name_tfidf",
    "addr_token_set", "addr_ratio", "addr_tfidf",
    "house_num", "name_missing", "addr_missing", "country_match",
    "script_mismatch", "domain_name", "cand_is_s3",
    *BLK_COLUMNS,
]

# Legal-form / filler words ignored by name_core_jaccard (French forms included for the unseen country).
_LEGAL = frozenset({
    "inc", "incorporated", "llc", "llp", "lp", "ltd", "limited", "corp", "corporation", "co",
    "company", "pvt", "private", "plc", "pllc", "pc", "the", "and", "of", "sarl", "sas", "sasu",
    "sa", "eurl", "sci", "snc", "groupe", "group", "gmbh", "dba", "et", "cie", "com", "www",
})
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
_LLC = re.compile(r"\bl l ([cp])\b")
_FIRST_NUM = re.compile(r"\d+")
_DOMAIN = re.compile(r"(^@)|(\.(com|net|org|in|co|fr|biz|io|us|info)\b)|(\bwww\b)", re.I)
_INDIC = re.compile(r"[ऀ-෿]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")

TFIDF_SAMPLE = 50_000  # pool rows used for IDF statistics (deterministic sample)
TEXT_BATCH = 10_000    # strings vectorized at a time (bounds transient n-gram memory)
PAIR_BLOCK = 20_000    # pairs per sparse-product block (bounds memory)
# Stateless char-trigram hashing (no vocabulary to store); 2^20 buckets keeps collisions negligible.
_HASHER = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2**20,
                            alternate_sign=False, norm=None, dtype=np.float32)


# ------------------------------------------------------------- normalization
def _canon(s: str, abbr: dict) -> str:
    s = _NON_WORD.sub(" ", _translit(s or "").lower().replace("&", " and "))
    s = _LLC.sub(r"ll\1", s)
    return " ".join(w for w in (abbr.get(t, t) for t in s.split()) if w)


def canon_name(values) -> list[str]:
    return [_canon(v, _NAME_ABBR) for v in values]


def canon_addr(values) -> list[str]:
    return [_canon(v, _ADDR_ABBR) for v in values]


def _script(s: str) -> str:
    if _INDIC.search(s):
        return "indic"
    return "latin" if (not s or _LATIN.search(s)) else "other"


def _col(r: pd.DataFrame, norm: str, raw: str) -> pd.Series:
    return (r[norm] if norm in r.columns else r[raw]).astype(str)


class _Side:
    """One side of the pairs: each distinct record processed once, plus per-pair codes."""

    def __init__(self, ids: np.ndarray, rec: pd.DataFrame, what: str):
        self.codes, uniq = pd.factorize(ids)
        if not rec.index.is_unique:
            raise ValueError(f"{what} records: entity_id index is not unique")
        pos = rec.index.get_indexer(uniq)
        if (pos < 0).any():
            missing = list(uniq[pos < 0][:5])
            raise KeyError(f"{int((pos < 0).sum())} {what} ids in `pairs` are not in the {what} records, "
                           f"e.g. {missing}. This is a join/data-integrity bug upstream.")
        r = rec.iloc[pos]
        raw_name = r["business_name"].astype(str).tolist()
        raw_addr = r["business_address"].astype(str).tolist()
        self.name = np.array(canon_name(_col(r, "name_norm", "business_name")), dtype=object)
        self.addr = np.array(canon_addr(_col(r, "addr_norm", "business_address")), dtype=object)
        if "house_no" in r.columns:
            house = r["house_no"].astype(str).str.lstrip("0").tolist()
        else:
            house = [(m.group(0).lstrip("0") if (m := _FIRST_NUM.search(a)) else "") for a in raw_addr]
        self.house = np.array(house, dtype=object)
        self.script = np.array(r["name_script"].astype(str).tolist() if "name_script" in r.columns
                               else [_script(x) for x in raw_name], dtype=object)
        self.domain = (r["name_is_domain"].astype(bool).to_numpy() if "name_is_domain" in r.columns
                       else np.array([bool(_DOMAIN.search(x)) for x in raw_name]))
        self.country = r["country"].astype(str).str.strip().str.lower().to_numpy(dtype=object)
        self.core = [frozenset(t for t in n.split() if t not in _LEGAL) for n in self.name]

    def take(self, arr):
        return arr[self.codes]


# ------------------------------------------------------------- similarities
def _hash(texts) -> sparse.csr_matrix:
    texts = list(texts)
    if not texts:
        return sparse.csr_matrix((0, _HASHER.n_features), dtype=np.float32)
    return sparse.vstack([_HASHER.transform(texts[i:i + TEXT_BATCH])
                          for i in range(0, len(texts), TEXT_BATCH)], format="csr")


def fit_vectorizers(pool: pd.DataFrame) -> dict:
    """IDF weights for char n-gram TF-IDF, from a deterministic sample of `pool` (static corpus
    statistics: the same for every S1 chunk of a country, independent of chunking/row order)."""
    idx = pool.index.sort_values()
    sample = pool.loc[idx[:: max(1, len(idx) // TFIDF_SAMPLE)]]
    names = canon_name(_col(sample, "name_norm", "business_name"))
    addrs = [a for a in canon_addr(_col(sample, "addr_norm", "business_address")) if a]
    return {key: TfidfTransformer().fit(_hash(docs or ["x"])) for key, docs in (("name", names), ("addr", addrs))}


def _tfidf_rows(tt: TfidfTransformer, texts) -> sparse.csr_matrix:
    """L2-normalised TF-IDF rows, one per distinct record, built batch by batch."""
    texts = list(texts)
    if not texts:
        return sparse.csr_matrix((0, _HASHER.n_features), dtype=np.float32)
    return sparse.vstack([tt.transform(_HASHER.transform(texts[i:i + TEXT_BATCH])).astype(np.float32)
                          for i in range(0, len(texts), TEXT_BATCH)], format="csr")


def _tfidf_cos(vec, a: _Side, b: _Side, field: str) -> np.ndarray:
    """Cosine of each pair: transform every distinct record once, then a sparse row-wise
    product over blocks of pairs (TF-IDF rows are already L2-normalised)."""
    xa = _tfidf_rows(vec, getattr(a, field))
    xb = _tfidf_rows(vec, getattr(b, field))
    n = len(a.codes)
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, PAIR_BLOCK):
        e = min(s + PAIR_BLOCK, n)
        prod = xa[a.codes[s:e]].multiply(xb[b.codes[s:e]])
        out[s:e] = np.asarray(prod.sum(axis=1)).ravel()
    return out


def _sim(scorer, x: list, y: list, both: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Pairwise similarity in C++ over all cores; 0 where either side is empty."""
    v = cpdist(x, y, scorer=scorer, workers=-1, dtype=np.float32) / scale
    return np.where(both, v, 0.0).astype(np.float32)


# ------------------------------------------------------------- main
def build_features(pairs: pd.DataFrame, s1: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """`pairs`: s1_id, cand_id, blk_*. `s1`, `pool`: normalized records indexed by entity_id.

    Returns float32 features, same index and order as `pairs`, columns == FEATURE_NAMES.
    Raises KeyError if a pair references an id missing from `s1` / `pool`.
    """
    n = len(pairs)
    if n == 0:
        return pd.DataFrame({c: pd.Series(dtype=np.float32) for c in FEATURE_NAMES}, index=pairs.index)
    a = _Side(pairs[S1_ID].to_numpy(), s1, "s1")
    b = _Side(pairs[CAND_ID].to_numpy(), pool, "pool")
    vec = fit_vectorizers(pool)

    an, bn = a.take(a.name), b.take(b.name)
    aa, ba = a.take(a.addr), b.take(b.addr)
    name_ok = (an != "") & (bn != "")
    addr_ok = (aa != "") & (ba != "")
    an, bn, aa, ba = an.tolist(), bn.tolist(), aa.tolist(), ba.tolist()

    f: dict[str, np.ndarray] = {}
    f["name_ratio"] = _sim(fuzz.ratio, an, bn, name_ok, 100)
    f["name_token_set"] = _sim(fuzz.token_set_ratio, an, bn, name_ok, 100)
    f["name_token_sort"] = _sim(fuzz.token_sort_ratio, an, bn, name_ok, 100)
    f["name_partial"] = _sim(fuzz.partial_ratio, an, bn, name_ok, 100)
    f["name_jaro_winkler"] = _sim(JaroWinkler.normalized_similarity, an, bn, name_ok)
    ca, cb = a.take(np.array(a.core, dtype=object)), b.take(np.array(b.core, dtype=object))
    f["name_core_jaccard"] = np.fromiter(
        ((len(x & y) / len(x | y)) if x and y else 0.0 for x, y in zip(ca, cb)), np.float32, n)
    f["name_tfidf"] = np.where(name_ok, _tfidf_cos(vec["name"], a, b, "name"), 0.0)

    f["addr_token_set"] = _sim(fuzz.token_set_ratio, aa, ba, addr_ok, 100)
    f["addr_ratio"] = _sim(fuzz.ratio, aa, ba, addr_ok, 100)
    f["addr_tfidf"] = np.where(addr_ok, _tfidf_cos(vec["addr"], a, b, "addr"), 0.0)

    ha, hb = a.take(a.house), b.take(b.house)
    both = (ha != "") & (hb != "")
    f["house_num"] = np.where(both, np.where(ha == hb, 1.0, -1.0), 0.0)
    f["name_missing"] = (~(a.take(a.name) != "")).astype(np.float32) + (b.take(b.name) == "")
    f["addr_missing"] = (~(a.take(a.addr) != "")).astype(np.float32) + (b.take(b.addr) == "")
    f["country_match"] = (a.take(a.country) == b.take(b.country)).astype(np.float32)
    f["script_mismatch"] = (a.take(a.script) != b.take(b.script)).astype(np.float32)
    f["domain_name"] = (a.take(a.domain) | b.take(b.domain)).astype(np.float32)
    f["cand_is_s3"] = pairs[CAND_ID].astype(str).str.startswith("S3-").to_numpy(np.float32)
    for c in BLK_COLUMNS:  # blocking's retrieval signals, passed through (NaN if absent)
        f[c] = (pd.to_numeric(pairs[c], errors="coerce").to_numpy(np.float32) if c in pairs.columns
                else np.full(n, np.nan, np.float32))

    out = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()},
                       index=pairs.index)[FEATURE_NAMES]
    return out.replace([np.inf, -np.inf], np.nan)
