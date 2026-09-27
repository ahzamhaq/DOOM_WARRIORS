"""Train, validate and save the matcher (Person 3). Run from the repo root:

    python -m scripts.build_dev_world            # once (Lead's frozen dev world)
    python -m notebooks.train_matcher            # train + threshold/one-to-one/per-country/LOCO report

Data: the frozen dev world (docs/CONTRACT.md §10), frozen train/val split. Its decoys are thinned
(~1/73 of real density), so by default it is topped up with "hard decoys": raw train S2/S3 records
of the same country sharing a name word with a dev-world S1 (at most --hard-decoys per word).
That makes precision numbers much less optimistic. --hard-decoys 0 = pure dev world.

Uses the real normalize.normalize_records / blocking.build_index + generate_candidates as soon as
they are implemented; until then features.py derives the normalized columns itself and a stand-in
retriever (country-level char TF-IDF on name and address, top BLOCK_K each, capped at
MAX_CANDS_PER_S1, emitting the same blk_* columns) produces the pairs. blocking.py is never
modified. Only the supplied training data is used.

Outputs (git-ignored): config.MODEL_PATH (model + tuned threshold), outputs/matcher_report.md,
outputs/threshold_sweep.tsv.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from src import blocking, normalize
from src.config import (BETA, BLK_COLUMNS, BLOCK_K, CAND_ID, MAX_CANDS_PER_S1, MODEL_PATH, ONE_TO_ONE,
                        OUTPUT_DIR, PROBA, PROCESSED_DIR, S1_ID, SOURCE_COLUMNS)
from src.data_loader import load_dev_world, source_path
from src.evaluate import truth_dict
from src.features import FEATURE_NAMES, _LEGAL, build_features, canon_name
from src.matcher import select_matches, train

T0 = time.time()
GRID = np.round(np.arange(0.10, 0.96, 0.05), 2)


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def _u(key: str) -> float:
    """Deterministic uniform [0,1) from a string (machine- and version-independent)."""
    return int(hashlib.sha1(key.encode()).hexdigest()[:12], 16) / 16 ** 12


# ------------------------------------------------------------- scoring
def f_beta(pred: set, truth: set, beta: float = BETA) -> float:
    """Per-entity F-beta exactly as in the README (singleton + empty prediction = 1.0)."""
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def score(matches: pd.DataFrame, truth: dict, s1_ids) -> dict:
    """Macro F0.5 over ALL `s1_ids` + pair-level precision/recall + singleton accuracy."""
    pred = matches.groupby(S1_ID)[CAND_ID].agg(set).to_dict() if len(matches) else {}
    tp = n_pred = n_true = single = single_ok = 0
    f = 0.0
    for s in s1_ids:
        p, t = pred.get(s, set()), truth.get(s, set())
        hit = len(p & t)
        tp, n_pred, n_true = tp + hit, n_pred + len(p), n_true + len(t)
        f += f_beta(p, t)
        if not t:
            single += 1
            single_ok += not p
    return {"precision": tp / n_pred if n_pred else 1.0, "recall": tp / n_true if n_true else 1.0,
            "singleton_acc": single_ok / single if single else 1.0,
            "macro_f05": f / len(s1_ids) if len(s1_ids) else 0.0, "pred_pairs": n_pred}


def sweep(scored: pd.DataFrame, truth: dict, s1_ids, one_to_one: bool) -> pd.DataFrame:
    return pd.DataFrame([{"threshold": float(t), **score(select_matches(scored, t, one_to_one), truth, s1_ids)}
                         for t in GRID])


def best(rep: pd.DataFrame) -> pd.Series:
    top = rep["macro_f05"].max()
    return rep[rep["macro_f05"] >= top - 1e-12].sort_values("threshold").iloc[-1]


# ------------------------------------------------------------- data
def stream_raw(source: int, keep) -> pd.DataFrame:
    reader = pacsv.open_csv(
        source_path("train", source),
        read_options=pacsv.ReadOptions(block_size=32 << 20),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in SOURCE_COLUMNS},
                                             include_columns=SOURCE_COLUMNS, strings_can_be_null=False))
    parts = []
    for batch in reader:
        df = batch.to_pandas()
        m = np.asarray(keep(df), dtype=bool)
        if m.any():
            parts.append(df[m])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SOURCE_COLUMNS)


def hard_decoys(s1: pd.DataFrame, exclude: set, per_token: int) -> pd.DataFrame:
    """Raw train S2/S3 records sharing a (non-legal) name word with a dev-world S1 of the same
    country, at most `per_token` per (country, word). Cached under data/processed/."""
    path = PROCESSED_DIR / f"hard_decoys_{per_token}.tsv"
    if path.exists():
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    words = defaultdict(set)
    for name, country in zip(canon_name(s1["business_name"]), s1["country"]):
        for w in name.split():
            if len(w) > 2 and w not in _LEGAL:
                words[w].add(country)
    used = defaultdict(int)

    def keep(df):
        m = np.zeros(len(df), dtype=bool)
        for i, (eid, name, country) in enumerate(zip(df["entity_id"], canon_name(df["business_name"]), df["country"])):
            if eid in exclude:
                continue
            for w in name.split():
                if country in words.get(w, ()) and used[(country, w)] < per_token:
                    used[(country, w)] += 1
                    m[i] = True
                    break
        return m

    out = pd.concat([stream_raw(src, keep) for src in (2, 3)], ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep="\t", index=False)
    return out


def normalize_or_raw(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return normalize.normalize_records(df)
    except NotImplementedError:
        return df  # features.py derives the normalized columns itself


# ------------------------------------------------------------- candidates
def _topk(q, c, k: int, batch: int = 200):
    ct = c.T.tocsr()
    for s in range(0, q.shape[0], batch):
        sim = (q[s:s + batch] @ ct).tocsr()
        for r in range(sim.shape[0]):
            lo, hi = sim.indptr[r], sim.indptr[r + 1]
            data, cols = sim.data[lo:hi], sim.indices[lo:hi]
            if len(data) > k:
                top = np.argpartition(-data, k - 1)[:k]
                data, cols = data[top], cols[top]
            order = np.argsort(-data, kind="stable")
            yield cols[order], data[order]


def standin_candidates(s1: pd.DataFrame, pool: pd.DataFrame, k: int = BLOCK_K,
                       max_cands: int = MAX_CANDS_PER_S1) -> pd.DataFrame:
    """Stand-in for blocking.generate_candidates (same output columns). One country at a time."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    frames = []
    for field, raw in (("name", "business_name"), ("addr", "business_address")):
        ctext = canon_name(pool[raw].astype(str))
        qtext = canon_name(s1[raw].astype(str))
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.01,
                              sublinear_tf=True, dtype=np.float32)
        try:
            cm = vec.fit_transform(ctext)
        except ValueError:
            continue
        qm = vec.transform(qtext)
        cids = pool.index.to_numpy()
        rows = [(sid, cids[j], float(v), r)
                for sid, (cols, vals) in zip(s1.index, _topk(qm, cm, k))
                for r, (j, v) in enumerate(zip(cols, vals)) if v > 0]
        frames.append(pd.DataFrame(rows, columns=[S1_ID, CAND_ID, f"blk_{field}_score", f"blk_{field}_rank"]))
    pairs = frames[0]
    for fr in frames[1:]:
        pairs = pairs.merge(fr, on=[S1_ID, CAND_ID], how="outer")
    for c in BLK_COLUMNS:
        if c not in pairs:
            pairs[c] = np.nan
    pairs[["blk_name_rank", "blk_addr_rank"]] = pairs[["blk_name_rank", "blk_addr_rank"]].fillna(-1)
    best_score = pairs[["blk_name_score", "blk_addr_score"]].max(axis=1)
    pairs = (pairs.assign(_b=best_score).sort_values([S1_ID, "_b"], ascending=[True, False])
                  .groupby(S1_ID, sort=False).head(max_cands).drop(columns="_b"))
    return pairs[[S1_ID, CAND_ID, *BLK_COLUMNS]].reset_index(drop=True)


def candidates(s1: pd.DataFrame, pool: pd.DataFrame, mode: str = "auto") -> tuple[pd.DataFrame, str]:
    """mode: auto = real blocking.py if implemented else stand-in; real = blocking.py (error if
    missing); standin = always the stand-in (baseline for comparing against real blocking)."""
    if mode != "standin":
        try:
            index = blocking.build_index(pool)
            return blocking.generate_candidates(s1, index), "blocking.py"
        except NotImplementedError:
            if mode == "real":
                raise
    note = "" if mode == "standin" else " (blocking.py not implemented yet)"
    return standin_candidates(s1, pool), "stand-in" + note


def blocking_report(data: pd.DataFrame, truth: dict, s1_country: pd.Series) -> pd.DataFrame:
    """Candidate recall by country and source (S2/S3) over ALL dev S1, plus per-retriever recall
    and candidates per S1: the target Person 2's blocking has to beat."""
    true = pd.DataFrame([(s, c) for s, cs in truth.items() for c in cs], columns=[S1_ID, CAND_ID])
    true["country"] = true[S1_ID].map(s1_country).to_numpy()
    true["source"] = true[CAND_ID].str[:2].to_numpy()
    found = data.loc[data["label"] == 1, [S1_ID, CAND_ID, "blk_name_rank", "blk_addr_rank"]].assign(_f=True)
    true = true.merge(found, on=[S1_ID, CAND_ID], how="left")
    true["found"] = true["_f"].eq(True)
    true["by_name"] = true["blk_name_rank"].fillna(-1) >= 0
    true["by_addr"] = true["blk_addr_rank"].fillna(-1) >= 0
    rows = []
    for (country, source), g in true.groupby(["country", "source"]):
        rows.append({"country": country, "source": source, "true_pairs": len(g), "recall": g["found"].mean(),
                     "recall_name_only": g["by_name"].mean(), "recall_addr_only": g["by_addr"].mean()})
    for country, g in true.groupby("country"):
        rows.append({"country": country, "source": "all", "true_pairs": len(g), "recall": g["found"].mean(),
                     "recall_name_only": g["by_name"].mean(), "recall_addr_only": g["by_addr"].mean()})
    rows.append({"country": "all", "source": "all", "true_pairs": len(true), "recall": true["found"].mean(),
                 "recall_name_only": true["by_name"].mean(), "recall_addr_only": true["by_addr"].mean()})
    rep = pd.DataFrame(rows)
    per_s1 = data.groupby(S1_ID).size()
    cps = {c: per_s1.reindex(s1_country.index[s1_country == c]).fillna(0) for c in rep["country"].unique() if c != "all"}
    cps["all"] = per_s1.reindex(s1_country.index).fillna(0)
    rep["cands_per_s1"] = rep["country"].map({c: v.mean() for c, v in cps.items()})
    rep["s1_without_cands"] = rep["country"].map({c: int((v == 0).sum()) for c, v in cps.items()})
    return rep


# ------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-decoys", type=int, default=60, help="hard decoys per name word (0 = none)")
    ap.add_argument("--blocking", choices=["auto", "real", "standin"], default="auto",
                    help="auto: real blocking.py if implemented, else the stand-in")
    args = ap.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dw = load_dev_world()
    truth = truth_dict(dw.truth_raw)
    s1_all, pool_all = dw.s1, dw.pool
    if args.hard_decoys:
        log(f"adding hard decoys ({args.hard_decoys} per name word, cached after the first run)")
        extra = hard_decoys(s1_all, set(pool_all.index), args.hard_decoys)
        extra = extra[~extra["entity_id"].isin(pool_all.index)].set_index("entity_id", drop=False)
        pool_all = pd.concat([pool_all, extra])
    log(f"dev world: {len(s1_all):,} S1, pool {len(pool_all):,} records")

    # ---- per country: candidates + features (a module never sees two countries at once) ----
    parts, source = [], ""
    for country in sorted(s1_all["country"].unique()):
        s1 = normalize_or_raw(s1_all[s1_all["country"] == country])
        pool = normalize_or_raw(pool_all[pool_all["country"] == country])
        pairs, source = candidates(s1, pool, args.blocking)
        feats = build_features(pairs, s1, pool)
        assert feats.index.equals(pairs.index) and list(feats.columns) == FEATURE_NAMES
        parts.append(pd.concat([pairs[[S1_ID, CAND_ID]].reset_index(drop=True), feats.reset_index(drop=True)],
                               axis=1).assign(country=country))
        log(f"{country}: {len(s1):,} S1, {len(pool):,} pool -> {len(pairs):,} pairs")
        del s1, pool, pairs, feats
        gc.collect()
    data = pd.concat(parts, ignore_index=True)
    del parts
    data["label"] = np.fromiter((c in truth.get(s, ()) for s, c in zip(data[S1_ID], data[CAND_ID])),
                                np.int8, len(data))
    split = s1_all["split"]
    data["split"] = data[S1_ID].map(split).to_numpy()
    # early stopping: 15 % of the TRAIN S1 entities (deterministic), validation stays untouched
    es = {s for s in s1_all.index[split == "train"] if _u("es" + s) < 0.15}
    data.loc[data[S1_ID].isin(es), "split"] = "es"

    n_true = sum(len(v) for v in truth.values())
    ceiling = data["label"].sum() / n_true
    blk_rep = blocking_report(data, truth, s1_all["country"])
    log(f"{len(data):,} pairs, recall ceiling {ceiling:.4f}, positives {data['label'].mean():.1%}")

    def fit(mask_tr, mask_es):
        return train(data.loc[mask_tr, FEATURE_NAMES], data.loc[mask_tr, "label"],
                     data.loc[mask_es, FEATURE_NAMES], data.loc[mask_es, "label"])

    def scored_of(model, mask):
        d = data.loc[mask]
        return pd.DataFrame({S1_ID: d[S1_ID].to_numpy(), CAND_ID: d[CAND_ID].to_numpy(),
                             PROBA: model.predict_proba(d[FEATURE_NAMES])})

    # ---- main model: train split -> validation split ----
    log("training on the train split")
    model = fit(data["split"] == "train", data["split"] == "es")
    val_ids = list(s1_all.index[split == "val"])
    scored_val = scored_of(model, data["split"] == "val")
    rep_off = sweep(scored_val, truth, val_ids, one_to_one=False)
    rep_on = sweep(scored_val, truth, val_ids, one_to_one=True)
    b_off, b_on = best(rep_off), best(rep_on)
    chosen = b_on if ONE_TO_ONE else b_off
    model.threshold = float(chosen["threshold"])
    model.save(MODEL_PATH)
    rep_off.to_csv(OUTPUT_DIR / "threshold_sweep.tsv", sep="\t", index=False, float_format="%.4f")
    # saved for notebooks/error_analysis.py (git-ignored outputs/)
    data.to_parquet(OUTPUT_DIR / "dev_pairs.parquet")
    scored_val.to_parquet(OUTPUT_DIR / "val_scored.parquet")

    # ---- per country at the chosen threshold ----
    rows = []
    for country in sorted(s1_all["country"].unique()):
        ids = [s for s in val_ids if s1_all.at[s, "country"] == country]
        sc = scored_val[scored_val[S1_ID].isin(set(ids))]
        for o2o in (False, True):
            rows.append({"country": country, "one_to_one": o2o, "S1": len(ids),
                         **score(select_matches(sc, model.threshold, o2o), truth, ids)})
    per_country = pd.DataFrame(rows)

    # ---- leave-one-country-out (proxy for the unseen country) ----
    loco, loco_reps = [], {}
    countries = sorted(s1_all["country"].unique())
    for held in countries:
        tr = (data["country"] != held) & (data["split"] == "train")
        es_m = (data["country"] != held) & (data["split"] == "es")
        m = fit(tr, es_m)
        ids = [s for s in val_ids if s1_all.at[s, "country"] == held]
        sc = scored_of(m, (data["country"] == held) & (data["split"] == "val"))
        rep = sweep(sc, truth, ids, one_to_one=False)
        loco_reps[held] = rep
        at_chosen = rep[rep["threshold"] == model.threshold].iloc[0]
        own = best(rep)
        loco.append({"train_on": "+".join(c for c in countries if c != held), "validate_on": held,
                     "f05_at_chosen_threshold": at_chosen["macro_f05"],
                     "best_threshold_there": own["threshold"], "best_f05_there": own["macro_f05"],
                     "one_to_one_on_at_chosen": score(select_matches(sc, model.threshold, True), truth, ids)["macro_f05"]})
    loco = pd.DataFrame(loco)

    # ---- threshold rule for an unseen country (France has no labels) ----
    # Each rule is scored on every held-out country of the LOCO runs; regret = that country's best
    # F0.5 minus the F0.5 the rule's threshold gets there. Pick the smallest worst-case regret.
    snap = lambda t: float(GRID[np.abs(GRID - t).argmin()])
    bests = loco["best_threshold_there"].to_numpy()
    rules = {"pooled validation best (current)": model.threshold,
             "mean of per-country bests": snap(bests.mean()),
             "stricter of per-country bests": float(bests.max()),
             "looser of per-country bests": float(bests.min())}
    rule_rows = []
    for name, t in rules.items():
        regrets = [loco_reps[h].loc[loco_reps[h]["threshold"] == t, "macro_f05"].iloc[0] for h in countries]
        regrets = [float(loco.loc[loco["validate_on"] == h, "best_f05_there"].iloc[0]) - f
                   for h, f in zip(countries, regrets)]
        rule_rows.append({"rule": name, "threshold": t, "worst_regret": max(regrets), "mean_regret": np.mean(regrets),
                          **{f"regret_{h}": r for h, r in zip(countries, regrets)}})
    rules_df = pd.DataFrame(rule_rows).sort_values(["worst_regret", "threshold"], ascending=[True, False])
    france = rules_df.iloc[0]

    imp = pd.Series(model.clf.booster_.feature_importance("gain"), index=model.feature_names)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    fmt = lambda df: "```\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n```"
    cols = ["threshold", "precision", "recall", "macro_f05", "singleton_acc", "pred_pairs"]
    both = rep_off[["threshold", "precision", "recall", "macro_f05"]].merge(
        rep_on[["threshold", "macro_f05"]].rename(columns={"macro_f05": "macro_f05_one_to_one"}), on="threshold")
    md = [
        "# Matcher validation report (Person 3)", "",
        f"- Data: frozen dev world, frozen split ({len(s1_all) - len(val_ids):,} train S1 incl. early-stopping "
        f"subset / {len(val_ids):,} validation S1); hard decoys per name word: {args.hard_decoys}",
        f"- Candidates: {source}; {len(data):,} pairs; recall ceiling {ceiling:.4f}",
        f"- Features: {len(FEATURE_NAMES)}; trees: {model.clf.best_iteration_ or model.clf.n_estimators}",
        f"- **Chosen threshold {model.threshold:.2f} (one_to_one={ONE_TO_ONE}) -> validation macro F0.5 "
        f"{chosen['macro_f05']:.4f}** (precision {chosen['precision']:.4f}, recall {chosen['recall']:.4f})",
        f"- Best with one_to_one=True: threshold {b_on['threshold']:.2f} -> {b_on['macro_f05']:.4f} "
        f"(difference {b_on['macro_f05'] - b_off['macro_f05']:+.4f})", "",
        "## Threshold -> precision -> recall -> macro F0.5 (one_to_one off)", "", fmt(rep_off[cols]), "",
        "## One-to-one off vs on", "", fmt(both), "",
        "## Per country at the chosen threshold", "", fmt(per_country[["country", "one_to_one", "S1", "precision",
                                                                         "recall", "macro_f05"]]), "",
        "## Leave-one-country-out (proxy for unseen France)", "", fmt(loco), "",
        "## Threshold rule for an unseen country (France)", "",
        f"Proposed: **{france['rule']} = {france['threshold']:.2f}** (smallest worst-case LOCO regret).", "",
        fmt(rules_df), "",
        "## Candidate recall (blocking target; all dev S1, both splits)", "", fmt(blk_rep), "",
        "## Feature importance (share of gain)", "", "```\n" + imp.to_string(float_format=lambda v: f"{v:.3f}") + "\n```",
    ]
    (OUTPUT_DIR / "matcher_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    log(f"saved model + threshold to {MODEL_PATH}")


if __name__ == "__main__":
    main()
