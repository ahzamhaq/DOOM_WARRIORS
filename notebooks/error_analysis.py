"""False-merge and missed-match analysis on the validation split (Person 3).

    python -m notebooks.train_matcher        # first: writes outputs/dev_pairs.parquet, val_scored.parquet
    python -m notebooks.error_analysis       # -> outputs/error_analysis.md (aggregates only)
                                             #    outputs/error_examples.tsv (records; git-ignored, never commit)

Types are assigned in a fixed priority order so each error is counted once; "gain if fixed" is the
validation macro F0.5 if every error of that type were removed (an upper bound on what a fix buys).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow.compute as pc

from src.config import CAND_ID, MODEL_PATH, OUTPUT_DIR, PROBA, S1_ID
from src.data_loader import load_dev_world, read_table, DATA_DIR
from src.evaluate import truth_dict
from src.features import _DOMAIN, _script, build_features
from src.matcher import Model, select_matches
from notebooks.train_matcher import hard_decoys, score


def assign(df: pd.DataFrame, rules: list[tuple[str, pd.Series]]) -> pd.Series:
    out = pd.Series("other", index=df.index, dtype=object)
    done = pd.Series(False, index=df.index)
    for name, mask in rules:
        m = mask & ~done
        out[m] = name
        done |= m
    return out


def table(df: pd.DataFrame, by: str, gain: dict | None = None) -> pd.DataFrame:
    t = df.groupby(by).size().rename("count").to_frame()
    t["share"] = t["count"] / t["count"].sum()
    for col in ("country", "source"):
        for v in sorted(df[col].unique()):
            t[f"{col}={v}"] = df[df[col] == v].groupby(by).size()
    if gain is not None:
        t["macro_f05_if_fixed"] = pd.Series(gain)
    return t.fillna(0).sort_values("count", ascending=False)


def owners(cand_ids: set) -> dict:
    """cand_id -> the S1 it truly belongs to (full train ground truth), for the given ids only."""
    gt = read_table(DATA_DIR / "train" / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"])
    gt = gt.filter(pc.not_equal(gt["matched_entity_ids"], "")).to_pandas()
    ex = gt.assign(c=gt["matched_entity_ids"].str.split(",")).explode("c")
    ex = ex[ex["c"].isin(cand_ids)]
    return dict(zip(ex["c"], ex["source1_entity_id"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-decoys", type=int, default=60, help="must match the train_matcher run")
    ap.add_argument("--threshold", type=float, default=None, help="default: the saved model's threshold")
    args = ap.parse_args()

    t = args.threshold if args.threshold is not None else Model.load(MODEL_PATH).threshold
    dw = load_dev_world()
    truth = truth_dict(dw.truth_raw)
    s1_all, pool = dw.s1, dw.pool
    if args.hard_decoys:
        extra = hard_decoys(s1_all, set(pool.index), args.hard_decoys)
        extra = extra[~extra["entity_id"].isin(pool.index)].set_index("entity_id", drop=False)
        pool = pd.concat([pool, extra])
    data = pd.read_parquet(OUTPUT_DIR / "dev_pairs.parquet")
    scored = pd.read_parquet(OUTPUT_DIR / "val_scored.parquet")
    val_ids = list(s1_all.index[s1_all["split"] == "val"])
    val = data[data[S1_ID].isin(set(val_ids))].merge(scored, on=[S1_ID, CAND_ID])
    val["source"] = val[CAND_ID].str[:2]
    matches = select_matches(scored, t, one_to_one=False)
    base = score(matches, truth, val_ids)
    md = [f"# Error analysis (validation split, threshold {t:.2f}, one_to_one off)", "",
          f"Validation macro F0.5 {base['macro_f05']:.4f}; precision {base['precision']:.4f}; "
          f"recall {base['recall']:.4f}; {len(val_ids):,} S1.", "",
          "Types are exclusive (first matching rule wins). `macro_f05_if_fixed` = F0.5 if every error of "
          "that type disappeared (upper bound on the gain from fixing it).", ""]

    # ------------------------------------------------ 1. false merges
    fp = val[(val[PROBA] >= t) & (val["label"] == 0)].copy()
    own = owners(set(fp[CAND_ID]))
    fp["cand_owner"] = np.where(fp[CAND_ID].map(own).isna(), "matches no S1 (decoy)", "belongs to another S1")
    fp["s1_is_singleton"] = fp[S1_ID].map(lambda s: not truth.get(s))
    hi, lo = 0.9, 0.7
    fp["type"] = assign(fp, [
        ("domain-style name", fp["domain_name"] == 1),
        ("script mismatch (Indic vs Latin)", fp["script_mismatch"] == 1),
        ("same address, different name", (fp["addr_token_set"] >= hi) & (fp["name_token_set"] < lo)),
        ("same name, different address", (fp["name_token_set"] >= hi) & (fp["addr_token_set"] < lo)),
        ("same name and address (near-duplicate)", (fp["name_token_set"] >= hi) & (fp["addr_token_set"] >= hi)),
        ("address missing", fp["addr_missing"] > 0),
        ("house number conflict", fp["house_num"] == -1),
    ])
    gain = {}
    for ty, g in fp.groupby("type"):
        drop = set(zip(g[S1_ID], g[CAND_ID]))
        kept = matches[[(s, c) not in drop for s, c in zip(matches[S1_ID], matches[CAND_ID])]]
        gain[ty] = score(kept, truth, val_ids)["macro_f05"]
    md += [f"## 1. False merges: {len(fp):,} wrong pairs out of {base['pred_pairs']:,} predicted", "",
           "```", table(fp, "type", gain).to_string(float_format=lambda v: f"{v:.4f}"), "```", "",
           "Who the wrongly matched record really belongs to:", "",
           "```", fp.groupby(["type", "cand_owner"]).size().unstack(fill_value=0).to_string(), "```", "",
           f"False merges on true singletons (each costs that S1 the full 1.0): "
           f"{int(fp['s1_is_singleton'].sum()):,} pairs on {fp.loc[fp['s1_is_singleton'], S1_ID].nunique():,} S1.", "",
           "Median similarities of the false merges vs the true matches that were found:", "", "```",
           pd.DataFrame({"false merges": fp[["name_token_set", "name_tfidf", "addr_token_set", "addr_tfidf", PROBA]].median(),
                         "true matches found": val[(val[PROBA] >= t) & (val["label"] == 1)][
                             ["name_token_set", "name_tfidf", "addr_token_set", "addr_tfidf", PROBA]].median()}
                        ).to_string(float_format=lambda v: f"{v:.3f}"), "```", ""]

    # ------------------------------------------------ 2. missed matches
    true = pd.DataFrame([(s, c) for s in val_ids for c in truth.get(s, ())], columns=[S1_ID, CAND_ID])
    true = true.merge(val[[S1_ID, CAND_ID, PROBA]], on=[S1_ID, CAND_ID], how="left")
    miss = true[true[PROBA].isna() | (true[PROBA] < t)].copy()
    miss["stage"] = np.where(miss[PROBA].isna(), "blocking (never a candidate)", "matcher (scored below threshold)")
    feats = pd.concat([build_features(g[[S1_ID, CAND_ID]], s1_all[s1_all["country"] == c], pool[pool["country"] == c])
                       for c, g in miss.assign(country=miss[S1_ID].map(s1_all["country"])).groupby("country")])
    miss = miss.join(feats)
    miss["country"] = miss[S1_ID].map(s1_all["country"])
    miss["source"] = miss[CAND_ID].str[:2]
    raw_a = s1_all.loc[miss[S1_ID], "business_name"].astype(str).to_numpy()
    raw_b = pool.loc[miss[CAND_ID], "business_name"].astype(str).to_numpy()
    indic = np.array([("indic" in (_script(x), _script(y))) for x, y in zip(raw_a, raw_b)])
    miss["type"] = assign(miss, [
        ("Indian-script name", pd.Series(indic, index=miss.index)),
        ("website/handle name", pd.Series([bool(_DOMAIN.search(x) or _DOMAIN.search(y)) for x, y in zip(raw_a, raw_b)],
                                          index=miss.index)),
        ("missing address", miss["addr_missing"] > 0),
        ("altered house number", miss["house_num"] == -1),
        ("name very different (token set < 0.5)", miss["name_token_set"] < 0.5),
        ("address very different (token set < 0.5)", miss["addr_token_set"] < 0.5),
    ])
    n_true = len(true)
    md += [f"## 2. Missed matches: {len(miss):,} of {n_true:,} true pairs "
           f"({(miss['stage'].str.startswith('blocking')).sum():,} never a candidate, "
           f"{(miss['stage'].str.startswith('matcher')).sum():,} scored below threshold)", "",
           "```", miss.groupby(["type", "stage"]).size().unstack(fill_value=0).assign(
               share_of_all_true=lambda d: d.sum(axis=1) / n_true).sort_values("share_of_all_true", ascending=False
           ).to_string(float_format=lambda v: f"{v:.4f}"), "```", "",
           "By country and source:", "", "```",
           miss.groupby(["country", "source", "stage"]).size().unstack(fill_value=0).to_string(), "```", ""]

    (OUTPUT_DIR / "error_analysis.md").write_text("\n".join(md) + "\n")
    ex = pd.concat([fp.assign(kind="false merge"), miss.assign(kind="missed match")])[
        ["kind", "type", S1_ID, CAND_ID, PROBA]]
    ex = ex.groupby(["kind", "type"]).head(15)
    ex["s1_name"] = s1_all.loc[ex[S1_ID], "business_name"].to_numpy()
    ex["s1_addr"] = s1_all.loc[ex[S1_ID], "business_address"].to_numpy()
    ex["cand_name"] = pool.loc[ex[CAND_ID], "business_name"].to_numpy()
    ex["cand_addr"] = pool.loc[ex[CAND_ID], "business_address"].to_numpy()
    ex.to_csv(OUTPUT_DIR / "error_examples.tsv", sep="\t", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
