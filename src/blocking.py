"""Candidate generation. Owner: Person 2.  Contract: docs/CONTRACT.md §4.

Everything here is already within ONE country (predict.py loops over countries). Two stages per S1:

  1. Shortlist (cheap, recall-first): word unigram+bigram TF-IDF cosine over name and address, via a hashed
     inverted index. Tokens found in more than _TOKEN_MAX_DF pool records are not indexed (they still count
     in the record's norm), so each lookup only walks short posting lists: no S1 x pool product.
     Score = name cosine + address cosine; the top _SHORTLIST pool records per S1 go on.
     Bigrams matter: address components get reordered, but words inside a component stay adjacent.
  2. Rank (the retrievers behind blk_*): char 3-gram TF-IDF cosine on name_norm and on addr_norm, computed
     only for the shortlisted pairs. Union of three top-k lists within the shortlist (name char, addr char,
     and the stage-1 score itself), capped at max_cands by best rank in any list.

Ranks are 0-based within a retriever; a retriever that did not return the pair gives score NaN, rank -1.
A pair that only the stage-1 list kept therefore has NaN/-1 in all four blk_* columns.
Ties everywhere: score desc, then cand_id asc (the pool is sorted by entity_id, so column order == id order).
Every step is row-local, so output does not depend on chunk size or pool row order.
Same code and parameters at train and test time, so blk_* features have the same distribution.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer

from src.config import BLK_COLUMNS, BLOCK_K, CAND_ID, MAX_CANDS_PER_S1, S1_ID

_STR = pd.ArrowDtype(pa.string())
_CHAR_NGRAM = (3, 3)
_TOKEN_MAX_DF = 1000
_SHORTLIST = 500
_QUERY_BATCH = 500
_FIELDS = (("name", "name_norm"), ("addr", "addr_norm"))
_TOKENS = HashingVectorizer(token_pattern=r"(?u)\S+", ngram_range=(1, 2), lowercase=False, alternate_sign=False,
                            norm=None, binary=True, n_features=2 ** 23, dtype=np.float32)  # stateless


_BUILD_SLICE = 200_000  # pool rows processed at a time while building: bounds the build's temporary memory


@dataclass
class _Field:
    tokens_t: sp.csr_matrix              # (hashed tokens x pool): cosine weights, frequent tokens dropped
    chars: TfidfVectorizer | None        # None when the pool has no text in this field
    char_matrix: sp.csr_matrix | None    # (pool x char n-grams), L2-normalized


@dataclass
class BlockIndex:
    """Whatever the retrievers need (vectorizers, sparse matrices). Built once per country."""
    ids: pa.Array             # pool entity_ids, sorted (Arrow: compact, no per-id Python objects)
    countries: pd.Index       # distinct pool country values
    country_code: np.ndarray  # per pool row: position in `countries`
    fields: dict              # "name" / "addr" -> _Field


def _slices(texts: pd.Series):
    for a in range(0, len(texts), _BUILD_SLICE):
        yield a, texts.iloc[a:a + _BUILD_SLICE]


def _token_index(texts: pd.Series) -> sp.csr_matrix:
    """Two passes over slices (df, then weights), so the full unpruned token matrix never exists.

    Per-row arithmetic is the original whole-matrix formula applied to a slice, so values are bit-identical.
    """
    n, n_tok = len(texts), _TOKENS.n_features
    df = np.zeros(n_tok, np.int64)
    for _, t in _slices(texts):
        df += np.bincount(_TOKENS.transform(t).indices, minlength=n_tok)
    idf = sp.diags((np.log((n + 1) / (df + 1)) + 1).astype(np.float32))
    keep = df <= _TOKEN_MAX_DF
    del df
    rows, cols, vals = [], [], []
    for a, t in _slices(texts):
        w = _TOKENS.transform(t) @ idf
        norm = np.sqrt(np.asarray(w.multiply(w).sum(1)).ravel())
        w = sp.diags((1 / np.where(norm > 0, norm, 1)).astype(np.float32)) @ w
        w = w.tocoo()
        m = keep[w.col]
        rows.append(w.row[m].astype(np.int32) + a), cols.append(w.col[m].astype(np.int32)), vals.append(w.data[m])
        del w, m
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    order = np.lexsort((rows, cols))  # token-major, pool rows ascending: the layout of the old `w.T.tocsr()`
    indptr = np.zeros(n_tok + 1, np.int32)
    np.cumsum(np.bincount(cols, minlength=n_tok), out=indptr[1:])
    return sp.csr_matrix((vals[order], rows[order], indptr), shape=(n_tok, n))


def _char_tfidf(texts: pd.Series) -> tuple[TfidfVectorizer, sp.csr_matrix]:
    """Same vectorizer and matrix as TfidfVectorizer(...).fit_transform(texts), built slice by slice.

    Mirrors scikit-learn 1.9's fit path exactly (tests/test_blocking.py checks it bit for bit):
    vocabulary ids in first-seen order, rows sorted by those ids, then renumbered alphabetically;
    float32 smoothed idf; per-slice sublinear tf, idf and L2 norm via the fitted TfidfTransformer.
    """
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=_CHAR_NGRAM, lowercase=False, sublinear_tf=True,
                          dtype=np.float32)
    analyze = vec.build_analyzer()
    first, df, nnz = {}, [], 0
    for _, t in _slices(texts):
        for doc in t.tolist():
            seen = set()
            for g in analyze(doc):
                j = first.get(g)
                if j is None:
                    j = first[g] = len(first)
                    df.append(0)
                seen.add(j)
            for j in seen:
                df[j] += 1
            nnz += len(seen)
    terms = sorted(first.items())
    renumber = np.empty(len(terms), np.int32)
    renumber[[old for _, old in terms]] = np.arange(len(terms), dtype=np.int32)
    d = np.empty(len(terms), np.float32)
    d[renumber] = df
    del df
    d += 1.0
    idf = np.full_like(d, fill_value=len(texts) + 1, dtype=np.float32)
    idf /= d
    np.log(idf, out=idf)
    idf += 1.0
    vec.vocabulary_ = {t: i for i, (t, _) in enumerate(terms)}
    vec.fixed_vocabulary_ = False
    vec._tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
    vec._tfidf.idf_, vec._tfidf.n_features_in_ = idf, len(idf)

    data, indices = np.empty(nnz, np.float32), np.empty(nnz, np.int32)
    indptr, at = np.zeros(len(texts) + 1, np.int32), 0
    for a, t in _slices(texts):
        cols, counts, ptr = [], [], [0]
        for doc in t.tolist():
            fc = {}
            for g in analyze(doc):
                j = first[g]
                fc[j] = fc.get(j, 0) + 1
            cols.extend(fc.keys())
            counts.extend(fc.values())
            ptr.append(len(cols))
        x = sp.csr_matrix((np.asarray(counts, np.intc), np.asarray(cols, np.int32), np.asarray(ptr, np.int32)),
                          shape=(len(ptr) - 1, len(terms)), dtype=np.float32)
        x.sort_indices()
        x.indices = renumber.take(x.indices)
        x = vec._tfidf.transform(x, copy=False)
        data[at:at + x.nnz], indices[at:at + x.nnz] = x.data, x.indices
        indptr[a + 1:a + 1 + x.shape[0]] = x.indptr[1:] + at
        at += x.nnz
        del x, cols, counts, ptr
    return vec, sp.csr_matrix((data, indices, indptr), shape=(len(texts), len(terms)))


def _field(texts: pd.Series) -> _Field:
    tokens_t = _token_index(texts)
    if not (texts != "").any():
        return _Field(tokens_t, None, None)
    return _Field(tokens_t, *_char_tfidf(texts))


def build_index(pool: pd.DataFrame) -> BlockIndex:
    """`pool`: normalized S2+S3 records of one country, indexed by entity_id."""
    ids = pa.array(pool["entity_id"].astype(str), type=pa.string())
    keep = np.flatnonzero(pc.or_(pc.starts_with(ids, "S2-"), pc.starts_with(ids, "S3-")).to_numpy(zero_copy_only=False))
    keep = keep[pc.sort_indices(ids.take(keep)).to_numpy()]  # ids are unique ASCII: byte order == str order
    codes, countries = pd.factorize(pool["country"].astype(str).iloc[keep], sort=True)
    return BlockIndex(
        ids=ids.take(keep),
        countries=pd.Index(countries),
        country_code=codes.astype(np.int32),
        fields={name: _field(pool[col].astype(str).iloc[keep].reset_index(drop=True)) for name, col in _FIELDS},
    )


def _rank_in_row(rows: np.ndarray, cols: np.ndarray, score: np.ndarray) -> np.ndarray:
    """0-based rank of each entry within its row (score desc, col asc). `rows` need not be sorted."""
    o = np.lexsort((cols, -score, rows))
    starts = np.r_[0, np.flatnonzero(np.diff(rows[o])) + 1]
    rank = np.empty(len(o), np.int64)
    rank[o] = np.arange(len(o)) - np.repeat(starts, np.diff(np.r_[starts, len(o)]))
    return rank


def _top_per_row(r: sp.csr_matrix, m: int):
    """Top-m entries per row of a score matrix -> (row, col, score). Partition first, so only survivors get sorted."""
    r = r.tocsr()
    r.sum_duplicates()
    ptr, cols, data = r.indptr, r.indices, r.data
    lens = np.diff(ptr)
    keep = data > 0
    for i in np.flatnonzero(lens > m):
        a, b = ptr[i], ptr[i + 1]
        d = data[a:b]
        keep[a:b] &= d >= np.partition(d, b - a - m)[b - a - m]  # ties with the m-th survive, cut below by col
    rows = np.repeat(np.arange(r.shape[0]), lens)[keep]
    cols, data = cols[keep], data[keep]
    sel = _rank_in_row(rows, cols, data) < m
    return rows[sel], cols[sel].astype(np.int64), data[sel]


def _char_cosine(q: sp.csr_matrix, p: sp.csr_matrix, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.asarray(q[rows].multiply(p[cols]).sum(1), dtype=np.float32).ravel()


def _batch(s1: pd.DataFrame, index: BlockIndex, k: int, max_cands: int):
    # Stage 1: shortlist
    score = None
    for name, col in _FIELDS:
        s = _TOKENS.transform(s1[col].astype(str)) @ index.fields[name].tokens_t
        score = s if score is None else score + s
    rows, cols, word = _top_per_row(score, _SHORTLIST)
    del score
    s1_code = index.countries.get_indexer(s1["country"].astype(str))  # -1: country absent from the pool
    same = s1_code[rows] == index.country_code[cols]
    rows, cols, word = rows[same], cols[same], word[same]

    # Stage 2: char TF-IDF retrievers on the shortlist
    word_rank = _rank_in_row(rows, cols, word)
    blk = {}
    for name, col in _FIELDS:
        f = index.fields[name]
        sc = (np.zeros(len(rows), np.float32) if f.chars is None
              else _char_cosine(f.chars.transform(s1[col].astype(str)).tocsr(), f.char_matrix, rows, cols))
        rk = _rank_in_row(rows, cols, sc)
        hit = (sc > 0) & (rk < k)
        blk[f"blk_{name}_score"] = np.where(hit, sc, np.nan).astype(np.float32)
        blk[f"blk_{name}_rank"] = np.where(hit, rk, -1).astype(np.int16)
    word_rank = np.where(word_rank < k, word_rank, -1)
    keep = (blk["blk_name_rank"] >= 0) | (blk["blk_addr_rank"] >= 0) | (word_rank >= 0)
    rows, cols, word_rank = rows[keep], cols[keep], word_rank[keep]
    blk = {c: v[keep] for c, v in blk.items()}

    # Cap after the union: best rank in any list, then best char score, then cand_id.
    ranks = np.stack([blk["blk_name_rank"], blk["blk_addr_rank"], word_rank]).astype(np.int64)
    best_rank = np.where(ranks < 0, np.iinfo(np.int64).max, ranks).min(0)
    best_score = np.nan_to_num(np.fmax(blk["blk_name_score"], blk["blk_addr_score"]), nan=-1.0)
    o = np.lexsort((cols, -best_score, best_rank, rows))
    rows_o = rows[o]
    starts = np.r_[0, np.flatnonzero(np.diff(rows_o)) + 1]
    o = o[np.arange(len(o)) - np.repeat(starts, np.diff(np.r_[starts, len(o)])) < max_cands]
    return rows[o], cols[o], {c: v[o] for c, v in blk.items()}


def generate_candidates(
    s1: pd.DataFrame, index: BlockIndex, k: int = BLOCK_K, max_cands: int = MAX_CANDS_PER_S1
) -> pd.DataFrame:
    """`s1`: normalized S1 chunk. Returns pairs: s1_id, cand_id + config.BLK_COLUMNS.

    Same country only, S2-/S3- ids only, no duplicate pairs, at most `max_cands` per S1.
    S1 entities with no candidates just have no rows.
    """
    s1_ids = s1["entity_id"].astype(str).to_numpy(object)
    parts = []
    for start in range(0, len(s1), _QUERY_BATCH):
        rows, cols, blk = _batch(s1.iloc[start:start + _QUERY_BATCH], index, k, max_cands)
        parts.append(pd.DataFrame({
            S1_ID: pd.array(s1_ids[start + rows], dtype=_STR),
            CAND_ID: pd.arrays.ArrowExtensionArray(index.ids.take(cols)),
            **{c: blk[c] for c in BLK_COLUMNS},
        }))
    if not parts:
        return pd.DataFrame({S1_ID: pd.array([], dtype=_STR), CAND_ID: pd.array([], dtype=_STR),
                             **{c: np.array([], np.int16 if c.endswith("_rank") else np.float32) for c in BLK_COLUMNS}})
    return pd.concat(parts, ignore_index=True).drop_duplicates([S1_ID, CAND_ID], ignore_index=True)
