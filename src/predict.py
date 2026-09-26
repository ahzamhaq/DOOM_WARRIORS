"""End-to-end pipeline driver. Owner: Lead.  Contract: docs/CONTRACT.md §1 and §8.

    python -m src.predict test --dry-run     # empty submission; passes the official validator today
    python -m src.predict test               # full test run -> outputs/matching_results.tsv + candidate_pairs.tsv

`run_pipeline` is the single integration path. The real run and tests/test_contract.py (fake components)
both go through it, so the toy test exercises exactly the code that produces the submission.
`train` and `dev` modes are added by the Lead once matcher.train exists.
Processes one country at a time, in S1 chunks; both output files are streamed to disk per country.
"""
import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from src import blocking, features, matcher, normalize
from src.config import (
    CAND_ID, CANDIDATE_FILE, CHUNK_S1, MATCHING_FILE, MODEL_PATH, ONE_TO_ONE, OUTPUT_DIR, PROBA,
    PROBA_FLOOR, S1_ID,
)
from src.data_loader import append_id_lists, list_countries, load_source, start_id_list_file

_NO_PAIRS = pd.DataFrame({S1_ID: [], CAND_ID: []})


@dataclass(frozen=True)
class Components:
    """The five module functions the pipeline calls. Swappable so tests can inject fakes."""
    normalize_records: Callable
    build_index: Callable
    generate_candidates: Callable
    build_features: Callable
    select_matches: Callable


REAL = Components(
    normalize.normalize_records, blocking.build_index, blocking.generate_candidates,
    features.build_features, matcher.select_matches,
)


def score_country(s1: pd.DataFrame, pool: pd.DataFrame, model, cand_path: Path,
                  comps: Components = REAL, chunk_s1: int = CHUNK_S1) -> pd.DataFrame:
    """Block, featurize and score every S1 chunk of one country (normalized inputs).

    Appends the candidate sets to `cand_path` as it goes. Returns scored pairs (s1_id, cand_id, proba)
    with proba >= PROBA_FLOOR.
    """
    index = comps.build_index(pool)
    parts = []
    for start in range(0, len(s1), chunk_s1):
        chunk = s1.iloc[start:start + chunk_s1]
        cands = comps.generate_candidates(chunk, index)
        append_id_lists(cands, chunk["entity_id"], cand_path)
        if cands.empty:
            continue
        proba = model.predict_proba(comps.build_features(cands, chunk, pool))
        keep = proba >= PROBA_FLOOR
        parts.append(cands.loc[keep, [S1_ID, CAND_ID]].assign(**{PROBA: proba[keep]}))
        del cands, proba
        gc.collect()
    return pd.concat(parts, ignore_index=True) if parts else _NO_PAIRS.assign(**{PROBA: []})


def run_pipeline(countries, load_country: Callable, model, out_dir: Path, comps: Components = REAL,
                 chunk_s1: int = CHUNK_S1, one_to_one: bool = ONE_TO_ONE, log: Callable = print) -> None:
    """Country by country: load -> normalize -> block -> features -> score -> select -> stream to disk.

    `load_country(country) -> (s1_raw, pool_raw)`: raw records indexed by entity_id; the pool is that
    country's S2 + S3 records concatenated.
    """
    match_path, cand_path = out_dir / MATCHING_FILE, out_dir / CANDIDATE_FILE
    start_id_list_file(match_path, "matched_entity_ids")
    start_id_list_file(cand_path, "candidate_entity_ids")
    for country in countries:
        s1_raw, pool_raw = load_country(country)
        s1, pool = comps.normalize_records(s1_raw), comps.normalize_records(pool_raw)
        scored = score_country(s1, pool, model, cand_path, comps, chunk_s1)
        matches = comps.select_matches(scored, model.threshold, one_to_one=one_to_one)
        append_id_lists(matches, s1["entity_id"], match_path)
        log(f"{country}: {len(s1):,} S1, {len(scored):,} scored pairs, {len(matches):,} matches")
        del s1, pool, scored, matches, s1_raw, pool_raw
        gc.collect()


def run_test(dry_run: bool = False) -> None:
    if dry_run:  # empty but valid submission: validator smoke test, needs no model
        match_path, cand_path = OUTPUT_DIR / MATCHING_FILE, OUTPUT_DIR / CANDIDATE_FILE
        start_id_list_file(match_path, "matched_entity_ids")
        start_id_list_file(cand_path, "candidate_entity_ids")
        for country in list_countries("test"):
            ids = load_source("test", 1, country)["entity_id"]
            append_id_lists(_NO_PAIRS, ids, match_path)
            append_id_lists(_NO_PAIRS, ids, cand_path)
        return

    def load_country(country):  # countries come from the data, never a hard-coded list
        pool = pd.concat([load_source("test", 2, country), load_source("test", 3, country)])
        return load_source("test", 1, country), pool

    run_pipeline(list_countries("test"), load_country, matcher.Model.load(MODEL_PATH), OUTPUT_DIR)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["test"])
    ap.add_argument("--dry-run", action="store_true", help="write an empty but valid submission")
    args = ap.parse_args()
    run_test(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
