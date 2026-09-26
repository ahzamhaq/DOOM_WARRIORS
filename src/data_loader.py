"""Loading source files, ground truth, and writing submission files.

All files are TAB-separated with no quoting. Strings are loaded as Arrow-backed
dtypes to keep memory low (full train+test is ~2.5 GB on disk). Missing fields
are empty strings, never NaN, so downstream code never has to special-case nulls.
"""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from src.config import (
    CANDIDATE_FILE, DATA_DIR, MATCHING_FILE, OUTPUT_DIR, SOURCE_COLUMNS,
)

_STR = pd.ArrowDtype(pa.string())


def _read_tsv(path: Path, columns: list[str]) -> pd.DataFrame:
    table = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in columns},
            include_columns=columns,
            strings_can_be_null=False,  # empty field -> "" rather than null
        ),
    )
    return table.to_pandas(types_mapper={pa.string(): _STR}.get)


def source_path(split: str, source: int) -> Path:
    return DATA_DIR / split / f"{split}_source{source}.tsv"


def load_source(split: str, source: int) -> pd.DataFrame:
    """Load one source file: entity_id, business_name, business_address, country."""
    return _read_tsv(source_path(split, source), SOURCE_COLUMNS)


def load_ground_truth_raw() -> pd.DataFrame:
    """Ground truth as provided: one row per S1 entity, comma-joined match list."""
    return _read_tsv(
        DATA_DIR / "train" / "train_ground_truth.tsv",
        ["source1_entity_id", "matched_entity_ids"],
    )


def load_ground_truth_pairs(gt: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ground truth exploded to positive pairs: (s1_id, match_id). Singletons dropped."""
    gt = load_ground_truth_raw() if gt is None else gt
    s = gt.set_index("source1_entity_id")["matched_entity_ids"].astype(str)
    pairs = s[s != ""].str.split(",").explode().reset_index()
    pairs.columns = ["s1_id", "match_id"]
    return pairs.astype(_STR)


def write_id_lists(mapping: dict[str, list[str]], s1_ids, path: Path, col: str) -> None:
    """Write a submission-style TSV: one row per S1 id (in s1_ids order), deduped ids."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s1 in s1_ids:
            ids = dict.fromkeys(mapping.get(s1, ()))  # dedupe, keep order
            f.write(f"{s1}\t{','.join(ids)}\n")


def write_submission(matches: dict, candidates: dict, s1_ids, out_dir: Path = OUTPUT_DIR) -> None:
    """Write both required output files for every S1 test entity."""
    write_id_lists(matches, s1_ids, out_dir / MATCHING_FILE, "matched_entity_ids")
    write_id_lists(candidates, s1_ids, out_dir / CANDIDATE_FILE, "candidate_entity_ids")
