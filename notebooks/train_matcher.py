"""Train + validate the pairwise matcher (Person 3). Run from the repo root:

    python3 -m notebooks.train_matcher                      # defaults
    python3 -m notebooks.train_matcher --candidates data/processed/train_candidates.tsv

Candidates, in order of preference:
  1. --candidates FILE : pairs produced by Person 2's blocking on train S1
     (columns s1_id, cand_id, optional rank/score; or the submission-style
     source1_entity_id / candidate_entity_ids format).
  2. src.blocking.generate_candidates, if it is implemented.
  3. A temporary stand-in (country-blocked char TF-IDF retrieval, the plan
     written in blocking.py) so the matcher can be built before blocking lands.
     It runs on a sample of train S1 against a reduced S2/S3 pool: all true
     matches of the sample, plus "hard" decoys that share a name word with a
     sampled S1 record, plus a random background slice. blocking.py is never
     modified.

Outputs (git-ignored): outputs/matcher_bundle.joblib, outputs/threshold_report.tsv,
outputs/matcher_report.md. Uses only the supplied training data.
"""
from __future__ import annotations

import argparse
import gc
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import OUTPUT_DIR, PROCESSED_DIR, SEED, SOURCE_COLUMNS
from src.data_loader import load_ground_truth_raw, load_source, source_path
from src.evaluate import candidate_recall, truth_dict
from src.features import FEATURE_COLUMNS, _LEGAL, build_features, fit_vectorizers, normalize_name
from src.matcher import (best_threshold, evaluate_matches, predict_proba, save_bundle,
                         select_matches, threshold_report, train)

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def stream_source(split: str, source: int, keep):
    """Read one source file in blocks, keeping rows where keep(df) is True."""
    reader = pacsv.open_csv(
        source_path(split, source),
        read_options=pacsv.ReadOptions(block_size=32 << 20),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in SOURCE_COLUMNS},
            include_columns=SOURCE_COLUMNS, strings_can_be_null=False),
    )
    parts = []
    for batch in reader:
        df = batch.to_pandas()
        m = np.asarray(keep(df), dtype=bool)
        if m.any():
            parts.append(df[m])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SOURCE_COLUMNS)


def core_tokens(norm_name: str) -> set:
    return {t for t in norm_name.split() if len(t) > 2 and t not in _LEGAL}


# ------------------------------------------------------------- stand-in -----
def build_pool(s1: pd.DataFrame, truth: dict, per_token: int, background: float) -> pd.DataFrame:
    """Reduced S2/S3 pool for the stand-in candidate generator."""
    needed = set().union(*(truth[s] for s in s1["entity_id"]))
    tok_country = defaultdict(set)  # token -> countries of sampled S1 using it
    for name, country in zip(normalize_name(s1["business_name"]), s1["country"]):
        for t in core_tokens(name):
            tok_country[t].add(country)
    used = defaultdict(int)  # (country, token) -> decoys kept so far
    rng = np.random.default_rng(SEED)

    def keep(df: pd.DataFrame) -> np.ndarray:
        m = np.array(df["entity_id"].isin(needed).to_numpy(), dtype=bool)  # writable copy
        m |= rng.random(len(df)) < background
        names = normalize_name(df["business_name"])
        for i, (name, country) in enumerate(zip(names, df["country"])):
            if m[i]:
                continue
            for t in core_tokens(name):
                if country in tok_country.get(t, ()) and used[(country, t)] < per_token:
                    used[(country, t)] += 1
                    m[i] = True
                    break
        return m

    parts = []
    for src in (2, 3):
        log(f"  streaming train_source{src}")
        parts.append(stream_source("train", src, keep))
    pool = pd.concat(parts, ignore_index=True)
    log(f"  pool: {len(pool):,} records ({int(pool['entity_id'].isin(needed).sum()):,} true matches)")
    return pool


def _topk(q, c, k: int, batch: int = 200):
    """Top-k cosine neighbours of rows of q among rows of c (both L2-normed)."""
    ct = c.T.tocsr()
    idx_all, sc_all = [], []
    for s in range(0, q.shape[0], batch):
        sim = (q[s:s + batch] @ ct).tocsr()
        for r in range(sim.shape[0]):
            row = sim.getrow(r)
            if row.nnz == 0:
                idx_all.append(np.empty(0, int)); sc_all.append(np.empty(0))
                continue
            kk = min(k, row.nnz)
            top = np.argpartition(-row.data, kk - 1)[:kk]
            order = top[np.argsort(-row.data[top])]
            idx_all.append(row.indices[order]); sc_all.append(row.data[order])
    return idx_all, sc_all


def standin_candidates(s1: pd.DataFrame, pool: pd.DataFrame, k: int) -> pd.DataFrame:
    """Country-blocked char TF-IDF retrieval on name and on name+address,
    top-k each, unioned; rank = best rank over the two retrievers."""
    rows = []
    for country, q in s1.groupby("country", sort=False):
        c = pool[pool["country"] == country].reset_index(drop=True)
        if c.empty:
            continue
        for fields in (["business_name"], ["business_name", "business_address"]):
            qt = normalize_name(q[fields].astype(str).agg(" ".join, axis=1))
            ctext = normalize_name(c[fields].astype(str).agg(" ".join, axis=1))
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2,
                                  max_df=0.01, sublinear_tf=True, dtype=np.float32)
            cm = vec.fit_transform(ctext)
            qm = vec.transform(qt)
            idx, sc = _topk(qm, cm, k)
            cids = c["entity_id"].to_numpy()
            for sid, ii, ss in zip(q["entity_id"], idx, sc):
                for r, (j, v) in enumerate(zip(ii, ss)):
                    rows.append((sid, cids[j], r, float(v)))
        log(f"  {country}: {len(q):,} S1 queries vs {len(c):,} pool records")
    cand = pd.DataFrame(rows, columns=["s1_id", "cand_id", "rank", "score"])
    return (cand.sort_values(["s1_id", "cand_id", "score"], ascending=[True, True, False])
                .groupby(["s1_id", "cand_id"], as_index=False)
                .agg(rank=("rank", "min"), score=("score", "max")))


def read_candidates(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "candidate_entity_ids" in df.columns:  # submission-style file
        df = (df.rename(columns={"source1_entity_id": "s1_id"})
                .assign(cand_id=lambda d: d["candidate_entity_ids"].str.split(","))
                .explode("cand_id"))
        df = df[df["cand_id"].fillna("") != ""][["s1_id", "cand_id"]]
    for col in ("rank", "score"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])
    return df.reset_index(drop=True)


# ------------------------------------------------------------- main ---------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", help="train candidate pairs TSV from blocking")
    ap.add_argument("--n-s1", type=int, default=20_000, help="train S1 sample (stand-in only)")
    ap.add_argument("--k", type=int, default=15, help="top-k per retriever (stand-in only)")
    ap.add_argument("--per-token", type=int, default=150, help="hard decoys per word (stand-in)")
    ap.add_argument("--background", type=float, default=0.01, help="random decoy share (stand-in)")
    ap.add_argument("--val-frac", type=float, default=0.3)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log("loading train S1 + ground truth")
    s1_all = load_source("train", 1)
    gt_raw = load_ground_truth_raw()

    source = "stand-in"
    if args.candidates:
        cands = read_candidates(args.candidates)
        source = f"blocking file {args.candidates}"
        s1 = s1_all[s1_all["entity_id"].isin(set(cands["s1_id"]))].reset_index(drop=True)
    else:
        s1 = s1_all.sample(n=min(args.n_s1, len(s1_all)), random_state=SEED).reset_index(drop=True)
        cands = None
    del s1_all
    gt_raw = gt_raw[gt_raw["source1_entity_id"].isin(set(s1["entity_id"]))]
    truth = truth_dict(gt_raw)  # only for the sampled entities (saves ~1 GB)
    del gt_raw
    gc.collect()
    log(f"{len(s1):,} train S1 entities, countries {s1['country'].value_counts().to_dict()}")

    if cands is not None:
        need = set(cands["cand_id"])
        pool = pd.concat([stream_source("train", s, lambda d: d["entity_id"].isin(need).to_numpy())
                          for s in (2, 3)], ignore_index=True)
    else:
        log("building reduced S2/S3 pool")
        pool = build_pool(s1, truth, args.per_token, args.background)
        try:
            from src.blocking import generate_candidates
            cands = generate_candidates(s1, pool)
            source = "src.blocking.generate_candidates (reduced pool)"
        except NotImplementedError:
            log("blocking.py not implemented yet -> using the temporary stand-in")
            cands = standin_candidates(s1, pool, args.k)

    cand_dict = cands.groupby("s1_id")["cand_id"].apply(list).to_dict()
    ceiling = candidate_recall(cand_dict, truth)
    log(f"candidates: {len(cands):,} pairs ({len(cands) / len(s1):.1f}/S1), recall ceiling {ceiling:.4f}")

    # ---- split by S1 entity (no entity in both train and validation) ----
    ids = s1["entity_id"].to_numpy()
    val_ids = set(rng.choice(ids, size=int(len(ids) * args.val_frac), replace=False))
    is_val = cands["s1_id"].isin(val_ids).to_numpy()
    cands["label"] = [int(c in truth[s]) for s, c in zip(cands["s1_id"], cands["cand_id"])]

    log("fitting TF-IDF vectorizers on train-side text")
    vecs = fit_vectorizers(s1[~s1["entity_id"].isin(val_ids)], pool)

    log("building features")
    feats = build_features(cands, s1, pool, vecs)
    X_tr, y_tr = feats[~is_val], cands.loc[~is_val, "label"]
    X_va, y_va = feats[is_val], cands.loc[is_val, "label"]
    log(f"train pairs {len(X_tr):,} (pos {y_tr.mean():.1%}) | val pairs {len(X_va):,} (pos {y_va.mean():.1%})")

    # Early stopping uses 15% of the *train* entities, so the validation
    # entities stay untouched until the threshold sweep.
    tr_ids = np.array(sorted(set(ids) - val_ids))
    es_ids = set(rng.choice(tr_ids, size=int(len(tr_ids) * 0.15), replace=False))
    is_es = cands.loc[~is_val, "s1_id"].isin(es_ids).to_numpy()
    log("training LightGBM")
    model = train(X_tr[~is_es], y_tr[~is_es], eval_set=(X_tr[is_es], y_tr[is_es]))
    proba = predict_proba(model, X_va)
    val_pairs = cands.loc[is_val, ["s1_id", "cand_id"]].reset_index(drop=True)
    val_truth = {s: truth[s] for s in val_ids}

    rep = threshold_report(val_pairs, proba, val_truth)
    rep_free = threshold_report(val_pairs, proba, val_truth, one_to_one=False)
    t_best = best_threshold(rep)
    best = rep[rep["threshold"] == t_best].iloc[0]

    base_all = evaluate_matches(val_pairs.groupby("s1_id")["cand_id"].apply(list).to_dict(), val_truth)
    rule = X_va["name_token_set"].to_numpy() >= 0.9
    base_rule = evaluate_matches(select_matches(val_pairs, rule.astype(float), 0.5), val_truth)
    base_empty = evaluate_matches({}, val_truth)

    save_bundle(OUTPUT_DIR / "matcher_bundle.joblib", model, vecs, t_best)
    rep.to_csv(OUTPUT_DIR / "threshold_report.tsv", sep="\t", index=False, float_format="%.4f")

    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURE_COLUMNS)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    cols = ["threshold", "precision", "recall", "macro_precision", "macro_recall",
            "singleton_acc", "macro_f05", "pred_pairs"]
    md = [
        "# Matcher validation report", "",
        f"- Candidates: {source}",
        f"- Train S1 entities: {len(s1) - len(val_ids):,} (15% of them for early stopping) | validation S1 entities: {len(val_ids):,}",
        f"- Candidate pairs: {len(cands):,} | recall ceiling (share of true pairs in candidates): {ceiling:.4f}",
        f"- Trees used (early stopping): {model.best_iteration_ or model.n_estimators}",
        f"- **Best threshold {t_best:.2f} -> macro F0.5 {best['macro_f05']:.4f}** "
        f"(precision {best['precision']:.4f}, recall {best['recall']:.4f})", "",
        "## Threshold sweep (one-to-one on)", "",
        "```\n" + rep[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n```", "",
        "## Same sweep without one-to-one clean-up", "",
        "```\n" + rep_free[["threshold", "precision", "recall", "macro_f05"]].to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n```", "",
        "## Baselines on the same validation entities", "",
        f"- Predict nothing: macro F0.5 {base_empty['macro_f05']:.4f}",
        f"- Predict every candidate: macro F0.5 {base_all['macro_f05']:.4f} "
        f"(precision {base_all['precision']:.4f})",
        f"- Rule name_token_set >= 0.9: macro F0.5 {base_rule['macro_f05']:.4f} "
        f"(precision {base_rule['precision']:.4f}, recall {base_rule['recall']:.4f})", "",
        "## Feature importance (share of gain)", "",
        "```\n" + imp.to_frame("gain").to_string(float_format=lambda v: f"{v:.3f}") + "\n```",
    ]
    (OUTPUT_DIR / "matcher_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    log("done")


if __name__ == "__main__":
    main()
