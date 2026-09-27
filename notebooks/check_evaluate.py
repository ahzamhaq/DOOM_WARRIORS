"""Cross-check evaluate.py on the frozen dev world (Person 4). Run from the repo root:

    python -m scripts.build_dev_world            # once
    python -m notebooks.check_evaluate           # ~same data + model as notebooks/train_matcher.py

Rebuilds exactly Person 3's validation setup (dev world + hard decoys, candidates from blocking.py or the
stand-in retriever, frozen split, same early-stopping subset and model), then checks that evaluate.py
gives the SAME numbers as train_matcher.py's own scoring code:
  * check_pairs / check_feats pass on every country's pairs and features;
  * blocking_report ALL/all recall == train_matcher's recall ceiling (PROGRESS.md: 0.9509 on stand-in blocking);
  * label_pairs == train_matcher labels; split_s1(dev S1 frame) == the frozen split column, and the hash
    recipe reproduces that column;
  * sweep_thresholds == train_matcher.sweep (one-to-one off and on), same chosen threshold
    (PROGRESS.md: 0.9400 at 0.70); per_country_report and loco_report next to Person 3's per-country / LOCO.
Any mismatch raises. Report: outputs/evaluate_check.md (git-ignored). Only the supplied training data is used.
"""
from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd

import notebooks.train_matcher as tm
from src import evaluate as ev
from src.config import CAND_ID, DEV_VAL_FRAC, OUTPUT_DIR, PROBA, S1_ID, SEED
from src.data_loader import load_dev_world
from src.features import FEATURE_NAMES, build_features
from src.matcher import select_matches, train


def same(a, b, what: str, atol: float = 1e-12) -> None:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=atol, equal_nan=True):
        raise AssertionError(f"MISMATCH {what}: evaluate.py {a[:8]} vs train_matcher {b[:8]}")
    tm.log(f"ok  {what}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-decoys", type=int, default=60, help="as train_matcher.py (0 = pure dev world)")
    ap.add_argument("--skip-train", action="store_true", help="only the blocking / label / split checks")
    args = ap.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dw = load_dev_world()
    truth = ev.truth_dict(dw.truth_raw)
    s1_all, pool_all = dw.s1, dw.pool
    if args.hard_decoys:
        extra = tm.hard_decoys(s1_all, set(pool_all.index), args.hard_decoys)
        extra = extra[~extra["entity_id"].isin(pool_all.index)].set_index("entity_id", drop=False)
        pool_all = pd.concat([pool_all, extra])
    tm.log(f"dev world: {len(s1_all):,} S1, pool {len(pool_all):,}")

    # ---- split: the frozen column is used as-is, and the hash recipe reproduces it ----
    tr, va = ev.split_s1(s1_all)
    assert set(va) == set(s1_all.index[s1_all["split"] == "val"]) and len(tr) + len(va) == len(s1_all)
    frozen_val = ev._u01(s1_all.index, 4, SEED) < DEV_VAL_FRAC  # salt 4 = scripts/build_dev_world SALT["split"]
    assert (frozen_val == (s1_all["split"] == "val").to_numpy()).all(), "frozen split recipe drifted"
    tm.log(f"ok  split_s1: frozen split used as-is ({len(tr):,} train / {len(va):,} val)")

    # ---- per country: candidates + contract checks + features (same loop as train_matcher) ----
    parts, source = [], ""
    for country in sorted(s1_all["country"].unique()):
        s1 = tm.normalize_or_raw(s1_all[s1_all["country"] == country])
        pool = tm.normalize_or_raw(pool_all[pool_all["country"] == country])
        pairs, source = tm.candidates(s1, pool)
        ev.check_pairs(pairs, s1, pool)
        feats = build_features(pairs, s1, pool)
        ev.check_feats(feats, pairs, FEATURE_NAMES)
        tm.log(f"ok  check_pairs + check_feats, {country}: {len(pairs):,} pairs")
        parts.append(pd.concat([pairs[[S1_ID, CAND_ID]], feats], axis=1).assign(country=country))
        del s1, pool, pairs, feats
        gc.collect()
    data = pd.concat(parts, ignore_index=True)
    del parts

    # ---- labels + blocking ----
    ref_label = np.fromiter((c in truth.get(s, ()) for s, c in zip(data[S1_ID], data[CAND_ID])), np.int8, len(data))
    data["label"] = ev.label_pairs(data, truth)
    same(data["label"], ref_label, "label_pairs")
    n_true = sum(len(v) for v in truth.values())
    blk = ev.blocking_report(data, truth, s1_all["country"])
    same(blk.set_index(["country", "source"]).loc[("ALL", "all"), "recall"], data["label"].sum() / n_true,
         "blocking_report recall == train_matcher recall ceiling")
    md = ["# evaluate.py cross-check (Person 4)", "",
          f"- Dev world + {args.hard_decoys} hard decoys per name word; candidates: {source}",
          f"- {len(data):,} pairs", "", ev_md("Blocking report (all dev S1)", blk)]
    if args.skip_train:
        return finish(md)

    # ---- model: exactly train_matcher's (train split, same early-stopping subset) ----
    split = s1_all["split"]
    data["split"] = data[S1_ID].map(split).to_numpy()
    es = {s for s in s1_all.index[split == "train"] if tm._u("es" + s) < 0.15}
    data.loc[data[S1_ID].isin(es), "split"] = "es"

    def fit(m_tr, m_es):
        return train(data.loc[m_tr, FEATURE_NAMES], data.loc[m_tr, "label"],
                     data.loc[m_es, FEATURE_NAMES], data.loc[m_es, "label"])

    def scored_of(model, mask):
        d = data.loc[mask]
        return pd.DataFrame({S1_ID: d[S1_ID].to_numpy(), CAND_ID: d[CAND_ID].to_numpy(),
                             PROBA: model.predict_proba(d[FEATURE_NAMES])})

    model = fit(data["split"] == "train", data["split"] == "es")
    val_ids = list(s1_all.index[split == "val"])
    scored_val = scored_of(model, data["split"] == "val")
    for o2o in (False, True):
        sel = lambda sc, t, o=o2o: select_matches(sc, t, one_to_one=o)  # noqa: E731
        ours = ev.sweep_thresholds(scored_val, truth, val_ids, sel, tm.GRID)
        ref = tm.sweep(scored_val, truth, val_ids, one_to_one=o2o)
        for col in ("threshold", "precision", "recall", "macro_f05", "singleton_acc", "pred_pairs"):
            same(ours[col], ref[col], f"sweep_thresholds.{col} (one_to_one={o2o})")
        same(ev._best_row(ours)["threshold"], tm.best(ref)["threshold"], f"chosen threshold (one_to_one={o2o})")
        if not o2o:
            sweep, best = ours, ev._best_row(ours)
    t = float(best["threshold"])
    tm.log(f"validation macro F0.5 {best['macro_f05']:.4f} at threshold {t:.2f} "
           f"(PROGRESS.md reproduction: 0.9400 at 0.70)")

    val_country = s1_all.loc[val_ids, "country"]
    per = ev.per_country_report(select_matches(scored_val, t), truth, val_country)
    for c in sorted(val_country.unique()):
        ids = list(val_country.index[val_country == c])
        ref = tm.score(select_matches(scored_val[scored_val[S1_ID].isin(set(ids))], t), truth, ids)
        same(per.set_index("country").loc[c, "macro_f05"], ref["macro_f05"], f"per_country_report {c}")

    def score_held_out(held):  # train_matcher's LOCO: fit on the other countries' train/es split
        others = data["country"] != held
        m = fit(others & (data["split"] == "train"), others & (data["split"] == "es"))
        return scored_of(m, (data["country"] == held) & (data["split"] == "val"))

    loco = ev.loco_report(score_held_out, truth, val_country, select_matches, tm.GRID, t)
    md += [f"- **Validation macro F0.5 {best['macro_f05']:.4f} at threshold {t:.2f}** "
           f"(precision {best['precision']:.4f}, recall {best['recall']:.4f})", "",
           ev_md("Threshold sweep (validation, one-to-one off)", sweep),
           ev_md("Per country at the chosen threshold", per),
           ev_md("Leave-one-country-out (proxy for unseen France)", loco)]
    finish(md)


def ev_md(title: str, df: pd.DataFrame) -> str:
    return f"## {title}\n\n```\n{df.to_string(index=False, float_format=lambda v: f'{v:.4f}')}\n```\n"


def finish(md: list[str]) -> None:
    path = OUTPUT_DIR / "evaluate_check.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)
    tm.log(f"all checks passed; report -> {path}")


if __name__ == "__main__":
    main()
