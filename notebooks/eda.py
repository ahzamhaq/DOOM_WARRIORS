"""Dataset EDA. Run from repo root:  python -m notebooks.eda

Memory-conscious: only one source file is held in memory at a time.
"""
import gc
import re

import numpy as np
import pandas as pd

from src.config import SEED
from src.data_loader import load_ground_truth_pairs, load_ground_truth_raw, load_source

pd.set_option("display.width", 200, "display.max_columns", 20)

DEVANAGARI = r"[ऀ-ॿ]"
NON_ASCII = r"[^\x00-\x7F]"
JUNK_PREFIX = r"^[^\wऀ-ॿ]+"          # e.g. "-- ", "<< "
DOMAIN = r"\.(?:com|net|org|in|co|fr|biz|io)\b"
US_ZIP = r"\b\d{5}(?:-\d{4})?\b"
IN_PIN = r"\b\d{3}\s?\d{3}\b"
POSTAL = r"\b\d{5,6}\b"
LEGAL = {
    "inc", "llc", "ltd", "limited", "corp", "corporation", "co", "company", "pvt", "private",
    "llp", "sarl", "sas", "sa", "sci", "eurl", "the", "and", "of", "&", "group", "groupe",
}


def header(t):
    print(f"\n{'=' * 90}\n{t}\n{'=' * 90}")


def profile(df: pd.DataFrame, name: str) -> None:
    header(f"{name}: {len(df):,} rows")
    print("dtypes:", dict(df.dtypes.astype(str)))
    print("duplicate entity_id:", int(df["entity_id"].duplicated().sum()))
    print("empty/whitespace per column:")
    for c in df.columns:
        print(f"   {c:18s} {int((df[c].str.strip() == '').sum()):>10,}")
    print("country:", df["country"].value_counts().to_dict())

    n, a = df["business_name"], df["business_address"]
    stats = pd.DataFrame({
        "name_len_med": n.str.len().groupby(df["country"]).median(),
        "addr_len_med": a.str.len().groupby(df["country"]).median(),
        "name_devanagari%": n.str.contains(DEVANAGARI).groupby(df["country"]).mean() * 100,
        "name_nonascii%": n.str.contains(NON_ASCII).groupby(df["country"]).mean() * 100,
        "name_junkprefix%": n.str.contains(JUNK_PREFIX).groupby(df["country"]).mean() * 100,
        "name_domain%": n.str.lower().str.contains(DOMAIN).groupby(df["country"]).mean() * 100,
        "addr_empty%": (a.str.strip() == "").groupby(df["country"]).mean() * 100,
        "addr_devanagari%": a.str.contains(DEVANAGARI).groupby(df["country"]).mean() * 100,
        "addr_has_5/6digit%": a.str.contains(POSTAL).groupby(df["country"]).mean() * 100,
    }).round(1)
    print(stats.to_string())
    print("sample rows:")
    print(df.sample(8, random_state=SEED).to_string(index=False, max_colwidth=70))


def postal(s: pd.Series) -> pd.Series:
    return s.str.extract(r"\b(\d{6}|\d{5})\b", expand=False).fillna("")


def name_tokens(s: str) -> set:
    toks = re.findall(r"[\wऀ-ॿ]+", s.lower())
    return {t for t in toks if t not in LEGAL and len(t) > 1}


def main():
    # ---------- 1-5: per-file profiling ----------
    s1_ids_train = None
    for split in ("train", "test"):
        for src in (1, 2, 3):
            df = load_source(split, src)
            profile(df, f"{split}_source{src}")
            if split == "train" and src == 1:
                s1_ids_train = set(df["entity_id"])
            del df
            gc.collect()

    # ---------- 6-7: ground truth ----------
    gt = load_ground_truth_raw()
    header(f"train_ground_truth: {len(gt):,} rows")
    print("dtypes:", dict(gt.dtypes.astype(str)))
    print("S1 ids in GT == S1 ids in train_source1:", set(gt["source1_entity_id"]) == s1_ids_train)
    lists = gt["matched_entity_ids"].astype(str)
    n_all = lists.map(lambda x: 0 if x == "" else x.count(",") + 1)
    n_s2 = lists.str.count("S2-")
    n_s3 = lists.str.count("S3-")
    bucket = pd.cut(n_all, [-1, 0, 1, 2, 3, 5, 10, 10**6], labels=["0", "1", "2", "3", "4-5", "6-10", ">10"])
    print("\n# matches per S1 entity:")
    print(pd.concat([bucket.value_counts().sort_index(),
                     (bucket.value_counts(normalize=True).sort_index() * 100).round(2)],
                    axis=1, keys=["count", "%"]).to_string())
    print(f"zero={int((n_all == 0).sum()):,}  one={int((n_all == 1).sum()):,}  multiple={int((n_all > 1).sum()):,}")
    print(f"matches per S1: mean={n_all.mean():.2f} median={n_all.median():.0f} max={n_all.max()}")
    print(f"S2 per S1: mean={n_s2.mean():.2f}  | S3 per S1: mean={n_s3.mean():.2f}")
    print("has S2 match %:", round((n_s2 > 0).mean() * 100, 2), " has S3 match %:", round((n_s3 > 0).mean() * 100, 2))

    pairs = load_ground_truth_pairs(gt)
    print(f"\ntotal positive pairs: {len(pairs):,}")
    dup = pairs["match_id"].duplicated().sum()
    print("S2/S3 ids matched to >1 S1 entity:", int(dup))
    matched_ids = set(pairs["match_id"])

    # sample of S1 entities for pair-level analysis
    rng = np.random.default_rng(SEED)
    sample_s1 = set(rng.choice(gt["source1_entity_id"].to_numpy(), size=20_000, replace=False))
    sp = pairs[pairs["s1_id"].isin(sample_s1)].copy()
    del gt, lists, n_all, n_s2, n_s3, bucket
    gc.collect()

    lookup = {}
    for src in (1, 2, 3):
        df = load_source("train", src)
        if src > 1:
            unmatched = (~df["entity_id"].isin(matched_ids)).sum()
            print(f"train_source{src}: {len(df):,} records, {unmatched:,} "
                  f"({unmatched / len(df) * 100:.1f}%) never matched to any S1 (distractors)")
        keep = df["entity_id"].isin(sample_s1 if src == 1 else set(sp["match_id"]))
        lookup[src] = df[keep].set_index("entity_id")
        del df
        gc.collect()
    del pairs, matched_ids
    gc.collect()

    # ---------- 8: what do true matches share? ----------
    s1 = lookup[1]
    other = pd.concat([lookup[2], lookup[3]])
    sp = sp.join(s1.add_prefix("a_"), on="s1_id").join(other.add_prefix("b_"), on="match_id")
    sp["src"] = sp["match_id"].str[:2]
    sp = sp.dropna(subset=["a_business_name", "b_business_name"])

    header(f"Positive-pair analysis on {len(sample_s1):,} sampled S1 entities ({len(sp):,} pairs)")
    sp["same_country"] = sp["a_country"] == sp["b_country"]
    pa_, pb_ = postal(sp["a_business_address"].astype(str)), postal(sp["b_business_address"].astype(str))
    both = (pa_ != "") & (pb_ != "")
    sp["postal_both"] = both
    sp["postal_eq"] = both & (pa_ == pb_)
    ta = sp["a_business_name"].astype(str).map(name_tokens)
    tb = sp["b_business_name"].astype(str).map(name_tokens)
    sp["name_tok_overlap"] = [len(x & y) > 0 for x, y in zip(ta, tb)]
    sp["name_jacc"] = [len(x & y) / max(1, len(x | y)) for x, y in zip(ta, tb)]
    sp["script_mismatch"] = (sp["a_business_name"].astype(str).str.contains(DEVANAGARI)
                             != sp["b_business_name"].astype(str).str.contains(DEVANAGARI))
    sp["b_addr_empty"] = sp["b_business_address"].astype(str).str.strip() == ""
    sp["exact_name"] = sp["a_business_name"].str.lower() == sp["b_business_name"].str.lower()

    g = sp.groupby(["a_country", "src"])
    summ = pd.DataFrame({
        "pairs": g.size(),
        "same_country%": g["same_country"].mean() * 100,
        "exact_name%": g["exact_name"].mean() * 100,
        "name_tok_overlap%": g["name_tok_overlap"].mean() * 100,
        "name_jacc_med": g["name_jacc"].median(),
        "script_mismatch%": g["script_mismatch"].mean() * 100,
        "b_addr_empty%": g["b_addr_empty"].mean() * 100,
        "postal_both%": g["postal_both"].mean() * 100,
        "postal_eq|both%": sp[both].groupby(["a_country", "src"])["postal_eq"].mean() * 100,
    }).round(1)
    print(summ.to_string())

    # blocking-key recall: fraction of positive pairs retained by each simple key
    header("Blocking-key recall on positive pairs (fraction of true pairs kept)")
    keys = {
        "same country": sp["same_country"],
        "share >=1 name token": sp["name_tok_overlap"],
        "same postal (when both have one)": sp.loc[both, "postal_eq"],
        "country & (name token OR postal eq)": sp["same_country"] & (sp["name_tok_overlap"] | sp["postal_eq"]),
    }
    for k, v in keys.items():
        print(f"   {k:45s} {v.mean() * 100:6.2f}%")

    header("Example positive pairs with NO shared name token (hard cases)")
    hard = sp[~sp["name_tok_overlap"]].sample(min(15, (~sp["name_tok_overlap"]).sum()), random_state=SEED)
    print(hard[["a_business_name", "b_business_name", "a_business_address", "b_business_address"]]
          .to_string(index=False, max_colwidth=45))
    header("Example positive pairs (random)")
    print(sp.sample(15, random_state=1)[["a_business_name", "b_business_name", "a_business_address", "b_business_address"]]
          .to_string(index=False, max_colwidth=45))


if __name__ == "__main__":
    main()
