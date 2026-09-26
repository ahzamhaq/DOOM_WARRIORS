"""Loading source files, ground truth, and writing submission files.

All files are TAB-separated with no quoting. Strings are loaded as Arrow-backed
dtypes to keep memory low (full train+test is ~2.5 GB on disk). Missing fields
are empty strings, never NaN, so downstream code never has to special-case nulls.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from src.config import (
    CANDIDATE_FILE, DATA_DIR, DEV_WORLD_DIR, MATCHING_FILE, OUTPUT_DIR, SOURCE_COLUMNS,
)

_STR = pd.ArrowDtype(pa.string())


def read_table(path: Path, columns: list[str]) -> pa.Table:
    """Read selected columns of a TSV as an all-string Arrow table (empty field -> "", never null)."""
    return pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        parse_options=pacsv.ParseOptions(delimiter="	", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in columns},
            include_columns=columns,
            strings_can_be_null=False,
        ),
    )


def _read_tsv(path: Path, columns: list[str], country: str | None = None) -> pd.DataFrame:
    table = read_table(path, columns)
    if country is not None:  # filter in Arrow, before the (larger) pandas conversion
        table = table.filter(pc.equal(table["country"], country))
    return table.to_pandas(types_mapper={pa.string(): _STR}.get)


def source_path(split: str, source: int) -> Path:
    return DATA_DIR / split / f"{split}_source{source}.tsv"


def load_source(split: str, source: int, country: str | None = None) -> pd.DataFrame:
    """Load one source file (optionally one country only).

    Returns records indexed by entity_id (also kept as a column), per docs/CONTRACT.md §2.
    """
    df = _read_tsv(source_path(split, source), SOURCE_COLUMNS, country)
    return df.set_index("entity_id", drop=False)


def list_countries(split: str) -> list[str]:
    """Countries present in a split's Source 1 file, sorted. Read from data, never hard-coded."""
    table = pacsv.read_csv(
        source_path(split, 1),
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(include_columns=["country"], column_types={"country": pa.string()}),
    )
    return sorted(pc.unique(table["country"]).to_pylist())


@dataclass
class DevWorld:
    """Frozen dev world (docs/CONTRACT.md §11). Records are indexed by entity_id like load_source()."""
    s1: pd.DataFrame          # SOURCE_COLUMNS + `split` ("train" / "val")
    s2: pd.DataFrame
    s3: pd.DataFrame
    truth_raw: pd.DataFrame   # raw ground-truth format; evaluate.truth_dict() turns it into a dict

    @property
    def pool(self) -> pd.DataFrame:
        return pd.concat([self.s2, self.s3])


def load_dev_world(country: str | None = None, split: str | None = None) -> DevWorld:
    """Load the frozen dev world, optionally one country and/or one split ("train" / "val") of S1."""
    d = DEV_WORLD_DIR
    if not (d / "s1.tsv").exists():
        raise FileNotFoundError(f"{d} missing. Build it: python -m scripts.build_dev_world")
    s1 = _read_tsv(d / "s1.tsv", SOURCE_COLUMNS + ["split"], country)
    if split is not None:
        s1 = s1[s1["split"] == split]
    s1 = s1.set_index("entity_id", drop=False)
    gt = _read_tsv(d / "ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"])
    gt = gt[gt["source1_entity_id"].isin(s1.index)].reset_index(drop=True)
    return DevWorld(
        s1=s1,
        s2=_read_tsv(d / "s2.tsv", SOURCE_COLUMNS, country).set_index("entity_id", drop=False),
        s3=_read_tsv(d / "s3.tsv", SOURCE_COLUMNS, country).set_index("entity_id", drop=False),
        truth_raw=gt,
    )


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


def start_id_list_file(path: Path, col: str) -> None:
    """Create/truncate a submission-style TSV and write its header (use before append_id_lists)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")


def append_id_lists(pairs: pd.DataFrame, s1_ids, path: Path) -> None:
    """Append one row per id in `s1_ids` from a pairs frame [s1_id, cand_id]; S1s with no pairs get empty rows.

    Streaming counterpart of write_id_lists: call once per chunk/country so results never sit in RAM.
    """
    joined = pairs.drop_duplicates(["s1_id", "cand_id"]).groupby("s1_id")["cand_id"].agg(",".join)
    lookup = joined.to_dict()
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.writelines(f"{s1}\t{lookup.get(s1, '')}\n" for s1 in s1_ids)


def write_submission(matches: dict, candidates: dict, s1_ids, out_dir: Path = OUTPUT_DIR) -> None:
    """Write both required output files for every S1 test entity."""
    write_id_lists(matches, s1_ids, out_dir / MATCHING_FILE, "matched_entity_ids")
    write_id_lists(candidates, s1_ids, out_dir / CANDIDATE_FILE, "candidate_entity_ids")
