"""Build the frozen dev world from the TRAIN files.  Owner: Lead.

    python -m scripts.build_dev_world [--n-s1 30000] [--update-manifest]

Deterministic recipe (fixed seed, independent of row order / pandas version / machine):
  1. u(id) = splitmix64(numeric part of id + seed-and-source-salt) / 2**64, uniform in [0, 1).
  2. S1: the n_s1 train S1 entities with the smallest u(id). Natural country mix is kept.
  3. Ground truth: their rows of train_ground_truth.
  4. S2/S3 pool: every record that truly matches a sampled S1 (all positives), PLUS every other
     S2/S3 record with u(id) < f, where f = n_s1 / (all train S1). "Other" = decoys and records of
     non-sampled S1 entities, so the true-match / confuser / decoy mix is the natural one, thinned by f.
  5. `split` column on S1: "val" if u_split(id) < DEV_VAL_FRAC else "train".
  6. All files sorted by entity_id, LF newlines, so the bytes (and sha256) are reproducible.
Output: data/processed/dev_world/{s1,s2,s3,ground_truth}.tsv + manifest.json (git-ignored; raw data
must never be committed). The manifest of hashes is committed as docs/dev_world_manifest.json so every
teammate can check that their regenerated copy is byte-identical.
"""
import argparse
import hashlib
import json
import platform
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from src.config import DATA_DIR, DEV_MANIFEST, DEV_N_S1, DEV_VAL_FRAC, DEV_WORLD_DIR, SEED, SOURCE_COLUMNS
from src.data_loader import read_table, source_path

SALT = {"S1": 1, "S2": 2, "S3": 3, "split": 4}
GT_COLS = ["source1_entity_id", "matched_entity_ids"]


def u01(ids: pa.ChunkedArray, salt: int, seed: int) -> np.ndarray:
    """Deterministic uniform [0,1) per id: splitmix64 of the numeric id part (ids look like 'S2-123456')."""
    num = pc.cast(pc.utf8_slice_codeunits(ids, 3), pa.uint64()).to_numpy()
    mix = (seed * 1_000_003 + salt) * 0x9E3779B97F4A7C15 % 2**64
    with np.errstate(over="ignore"):
        x = num + np.uint64(mix)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return x.astype(np.float64) / 2.0**64


def write_tsv(df: pd.DataFrame, path) -> None:
    """Plain TSV, LF newlines, no quoting (matches how the raw files are read)."""
    for c in df.columns:
        assert not df[c].str.contains("[\t\n\r]", regex=True).any(), f"tab/newline inside {c}"
    lines = df.iloc[:, 0]
    for c in df.columns[1:]:
        lines = lines + "\t" + df[c]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(df.columns) + "\n")
        f.write("\n".join(lines))
        f.write("\n")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(n_s1: int, seed: int) -> dict:
    out = DEV_WORLD_DIR
    out.mkdir(parents=True, exist_ok=True)

    # 1-2. sample S1
    t1 = read_table(source_path("train", 1), SOURCE_COLUMNS)
    n_total = t1.num_rows
    frac = n_s1 / n_total
    pick = np.sort(np.argsort(u01(t1["entity_id"], SALT["S1"], seed), kind="stable")[:n_s1])
    t1 = t1.take(pa.array(pick))
    s1 = t1.to_pandas()
    is_val = u01(t1["entity_id"], SALT["split"], seed) < DEV_VAL_FRAC
    s1["split"] = np.where(is_val, "val", "train")
    s1 = s1.sort_values("entity_id", kind="stable").reset_index(drop=True)
    sampled = set(s1["entity_id"])
    del t1

    # 3. ground truth of sampled S1
    gt = read_table(DATA_DIR / "train" / "train_ground_truth.tsv", GT_COLS).to_pandas()
    gt = gt[gt["source1_entity_id"].isin(sampled)].sort_values("source1_entity_id").reset_index(drop=True)
    matched = set(gt["matched_entity_ids"][gt["matched_entity_ids"] != ""].str.split(",").explode())

    # 4. S2/S3 pool: all positives + fraction f of everything else
    pools, stats = {}, {}
    for src in (2, 3):
        t = read_table(source_path("train", src), SOURCE_COLUMNS)
        is_pos = pc.is_in(t["entity_id"], value_set=pa.array(sorted(matched)))
        keep = np.asarray(is_pos) | (u01(t["entity_id"], SALT[f"S{src}"], seed) < frac)
        pools[src] = t.filter(pa.array(keep)).to_pandas().sort_values("entity_id", kind="stable").reset_index(drop=True)
        stats[f"S{src}_raw_rows"] = t.num_rows
        del t
    missing = matched - set(pools[2]["entity_id"]) - set(pools[3]["entity_id"])
    assert not missing, f"{len(missing)} ground-truth ids missing from pool"

    # 6. write
    files = {"s1.tsv": s1, "s2.tsv": pools[2], "s3.tsv": pools[3], "ground_truth.tsv": gt}
    for name, df in files.items():
        write_tsv(df, out / name)

    n_pos = int(sum(len(v.split(",")) for v in gt["matched_entity_ids"] if v))
    pool_all = pd.concat([pools[2], pools[3]])
    manifest = {
        "version": 1, "seed": seed, "n_s1": n_s1, "val_frac": DEV_VAL_FRAC,
        "sampling_fraction_f": round(frac, 6), "train_s1_total": n_total,
        "recipe": "see docstring of scripts/build_dev_world.py",
        "rows": {n: len(d) for n, d in files.items()},
        "s1_by_country": s1["country"].value_counts().sort_index().to_dict(),
        "s1_by_split": s1["split"].value_counts().sort_index().to_dict(),
        "pool_by_source_country": {
            f"S{k}/{c}": int(n) for k in (2, 3) for c, n in pools[k]["country"].value_counts().sort_index().items()
        },
        "positive_pairs": n_pos,
        "singleton_s1": int((gt["matched_entity_ids"] == "").sum()),
        "pool_records_matching_a_sampled_s1": len(matched),
        "pool_records_other": len(pool_all) - len(matched),
        "sha256": {n: sha256(out / n) for n in files},
        "env_info_not_verified": {"python": platform.python_version(), "pandas": pd.__version__,
                                  "numpy": np.__version__, "pyarrow": pa.__version__},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del stats
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-s1", type=int, default=DEV_N_S1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--update-manifest", action="store_true", help="overwrite the committed docs/ manifest")
    args = ap.parse_args()

    m = build(args.n_s1, args.seed)
    print(json.dumps({k: v for k, v in m.items() if k != "sha256"}, indent=2, sort_keys=True))

    if args.update_manifest or not DEV_MANIFEST.exists():
        DEV_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        DEV_MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {DEV_MANIFEST}")
        return 0
    ref = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    if (ref["seed"], ref["n_s1"]) != (m["seed"], m["n_s1"]):
        print("NOTE: seed/n_s1 differ from the committed manifest; not comparing hashes.")
        return 0
    bad = [n for n, h in m["sha256"].items() if ref["sha256"].get(n) != h]
    print("MATCHES committed manifest: byte-identical dev world." if not bad else f"MISMATCH vs manifest: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
