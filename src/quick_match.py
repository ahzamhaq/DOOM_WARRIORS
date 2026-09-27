"""Fast high-precision fallback matcher (deadline submission). Owner: Lead.

    python -m src.quick_match dev    # macro F0.5 on the frozen dev world (seconds)
    python -m src.quick_match test   # outputs/matching_results.tsv + candidate_pairs.tsv (minutes)

Rule: an S2/S3 record matches an S1 entity when, within the same country, the cleaned business name
(lowercased, punctuation and legal-form words removed) AND the first house number of the address are
identical. Keys shared by more than MAX_POOL_PER_KEY pool records are dropped (too ambiguous).
Precision-first, which suits macro F0.5; S1 entities without an exact key match get an empty list.
All string work runs in pyarrow (C++), one source file at a time.
"""
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from src.config import CAND_ID, CANDIDATE_FILE, DATA_DIR, MATCHING_FILE, OUTPUT_DIR, S1_ID, SOURCE_COLUMNS
from src.data_loader import append_id_lists, read_table, source_path, start_id_list_file

MAX_POOL_PER_KEY = 10
_LEGAL = (r"(^| )(private|pvt|limited|ltd|llc|l l c|llp|pllc|plc|inc|incorporated|corp|corporation|co|company|"
          r"pc|sarl|sas|sasu|eurl|sa|sci|snc|gmbh|the)( |$)")


def keys(t: pa.Table) -> pd.DataFrame:
    """entity_id, country, name key, house number (rows with an empty key or no number are dropped)."""
    n = pc.utf8_lower(t["business_name"])
    n = pc.replace_substring_regex(n, "&", " and ")
    n = pc.replace_substring_regex(n, r"[^\p{L}\p{N}]+", " ")
    for _ in range(2):  # legal words can be adjacent ("pvt ltd")
        n = pc.replace_substring_regex(n, _LEGAL, " ")
    n = pc.utf8_trim_whitespace(pc.replace_substring_regex(n, r"\s+", " "))
    h = pc.struct_field(pc.extract_regex(t["business_address"], r"0*(?P<h>[1-9][0-9]*)"), "h")
    df = pa.table({"entity_id": t["entity_id"], "country": t["country"], "nk": n, "house": h}).to_pandas()
    return df[(df["nk"] != "") & df["house"].notna()]


def match(s1: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    on = ["country", "nk", "house"]
    size = pool.groupby(on).size()
    ok = size[size <= MAX_POOL_PER_KEY].index
    pool = pool.set_index(on).loc[lambda d: d.index.isin(ok)].reset_index()
    pairs = s1.merge(pool, on=on, suffixes=("_s1", "_c"))
    return pd.DataFrame({S1_ID: pairs["entity_id_s1"], CAND_ID: pairs["entity_id_c"]}).drop_duplicates()


def run_test() -> None:
    s1_table = read_table(source_path("test", 1), SOURCE_COLUMNS)
    s1_ids = s1_table["entity_id"].to_pylist()
    s1 = keys(s1_table)
    del s1_table
    pool = pd.concat([keys(read_table(source_path("test", s), SOURCE_COLUMNS)) for s in (2, 3)], ignore_index=True)
    pairs = match(s1, pool)
    for f, col in ((MATCHING_FILE, "matched_entity_ids"), (CANDIDATE_FILE, "candidate_entity_ids")):
        start_id_list_file(OUTPUT_DIR / f, col)
        append_id_lists(pairs, s1_ids, OUTPUT_DIR / f)
    print(f"{len(s1_ids):,} S1, {pairs[S1_ID].nunique():,} with matches, {len(pairs):,} pairs -> {OUTPUT_DIR}")


def run_dev() -> None:
    from src.data_loader import load_dev_world
    from src.evaluate import macro_f_beta, truth_dict
    w = load_dev_world()
    tbl = lambda d: pa.Table.from_pandas(d[SOURCE_COLUMNS].astype(str), preserve_index=False)  # noqa: E731
    pairs = match(keys(tbl(w.s1)), keys(tbl(w.pool)))
    truth = truth_dict(w.truth_raw)
    pred = pairs.groupby(S1_ID)[CAND_ID].agg(set).to_dict()
    tp = sum(len(p & truth.get(s, set())) for s, p in pred.items())
    print(f"dev world: macro F0.5 {macro_f_beta(pred, truth):.4f}; pair precision {tp / max(1, len(pairs)):.4f}; "
          f"pair recall {tp / sum(len(t) for t in truth.values()):.4f}; S1 with a prediction {len(pred):,}/{len(truth):,}")


if __name__ == "__main__":
    {"dev": run_dev, "test": run_test}[sys.argv[1]]()
