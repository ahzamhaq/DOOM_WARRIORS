"""Benchmark features.build_features on one country's dev-world pairs (Person 3).

    python -m notebooks.bench_features --country India

Pairs come from the same stand-in retriever as notebooks/train_matcher.py (dev world + hard decoys),
cached to data/processed/bench_pairs_<country>.parquet so repeated runs time build_features only.
Reports wall time, µs/pair and peak RSS increase during the call (sampled every 20 ms).
"""
import argparse
import threading
import time

import pandas as pd

from src.config import PROCESSED_DIR
from src.data_loader import load_dev_world
from src.features import build_features


def rss_mb() -> float:
    with open("/proc/self/status") as fh:  # Linux
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def rss_mb_portable() -> float:
    try:
        return rss_mb()
    except OSError:  # macOS: current RSS via psutil if available
        try:
            import psutil
            return psutil.Process().memory_info().rss / 2**20
        except ImportError:
            return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--hard-decoys", type=int, default=60)
    args = ap.parse_args()

    from notebooks.train_matcher import hard_decoys, standin_candidates
    dw = load_dev_world(args.country)
    pool = dw.pool
    if args.hard_decoys:
        full = load_dev_world()
        extra = hard_decoys(full.s1, set(full.pool.index), args.hard_decoys)
        extra = extra[(extra["country"] == args.country) & ~extra["entity_id"].isin(pool.index)]
        pool = pd.concat([pool, extra.set_index("entity_id", drop=False)])
    s1 = dw.s1
    cache = PROCESSED_DIR / f"bench_pairs_{args.country}_{args.hard_decoys}.parquet"
    if cache.exists():
        pairs = pd.read_parquet(cache)
    else:
        pairs = standin_candidates(s1, pool)
        pairs.to_parquet(cache)
    print(f"{args.country}: {len(s1):,} S1, pool {len(pool):,}, pairs {len(pairs):,}")

    base = rss_mb_portable()
    peak = [base]
    done = threading.Event()

    def watch():
        while not done.is_set():
            peak[0] = max(peak[0], rss_mb_portable())
            time.sleep(0.02)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    t0 = time.perf_counter()
    feats = build_features(pairs, s1, pool)
    dt = time.perf_counter() - t0
    done.set()
    t.join()
    print(f"build_features: {dt:.1f} s for {len(pairs):,} pairs = {dt / len(pairs) * 1e6:.1f} µs/pair; "
          f"peak RSS increase {(peak[0] - base) / 1024:.2f} GB; output {feats.shape}")


if __name__ == "__main__":
    main()
