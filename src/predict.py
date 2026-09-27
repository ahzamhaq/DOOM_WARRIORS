"""End-to-end pipeline driver. Owner: Lead.  Contract: docs/CONTRACT.md §1 and §8.

    python -m src.predict train [--country C] [--limit-s1 N]   # dev world: train, tune threshold, save model
    python -m src.predict dev   [--country C] [--limit-s1 N]   # dev world: score validation S1 with saved model
    python -m src.predict test --dry-run                       # empty submission; passes the official validator
    python -m src.predict test                                 # full test run -> outputs/*.tsv

All three modes run the SAME per-country chunk loop (`_country_chunks`): normalize -> build_index +
build_context once per country -> per S1 chunk generate_candidates -> build_features. So the model is
trained on exactly the candidates and features it will see at test time. The module functions are
injected (`Components`, `Scoring`), so tests/test_contract.py and tests/test_predict_modes.py run the
real driver with fakes. Processes one country at a time; submission files are streamed to disk.
"""
import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from src import blocking, evaluate, features, matcher, normalize
from src.config import (
    CAND_ID, CANDIDATE_FILE, CHUNK_S1, ES_FRAC, MATCHING_FILE, MODEL_PATH, ONE_TO_ONE, OUTPUT_DIR, PROBA,
    PROBA_FLOOR, S1_ID, THRESHOLD_GRID,
)
from src.data_loader import (
    DevWorld, append_id_lists, list_countries, load_dev_world, load_source, start_id_list_file,
)

_NO_PAIRS = pd.DataFrame({S1_ID: [], CAND_ID: []})


@dataclass(frozen=True)
class Components:
    """The module functions the pipeline calls. Swappable so tests can inject fakes.

    `build_feature_context` is optional (CONTRACT §5, §8): when set it runs once per country and its
    result is passed to build_features(..., ctx=...); when None, build_features is called without ctx.
    `train_model` is only needed by the `train` mode.
    """
    normalize_records: Callable
    build_index: Callable
    generate_candidates: Callable
    build_features: Callable
    select_matches: Callable
    build_feature_context: Callable | None = None
    train_model: Callable | None = None


REAL = Components(
    normalize.normalize_records, blocking.build_index, blocking.generate_candidates,
    features.build_features, matcher.select_matches, features.build_context, matcher.train,
)


@dataclass(frozen=True)
class Scoring:
    """evaluate.py functions used by `train` / `dev` (CONTRACT §7). Swappable for tests."""
    split_s1: Callable
    label_pairs: Callable
    sweep_thresholds: Callable
    blocking_report: Callable
    per_country_report: Callable


REAL_SCORING = Scoring(evaluate.split_s1, evaluate.label_pairs, evaluate.sweep_thresholds,
                       evaluate.blocking_report, evaluate.per_country_report)


# ------------------------------------------------------------------ the shared chunk loop
def _country_chunks(s1: pd.DataFrame, pool: pd.DataFrame, comps: Components, chunk_s1: int,
                    cand_path: Path | None = None):
    """Yield (candidate pairs, features) per S1 chunk of ONE country (normalized inputs).

    Appends every chunk's candidate set to `cand_path` when given. Chunks with no candidates are not yielded.
    """
    index = comps.build_index(pool)
    ctx = comps.build_feature_context(pool) if comps.build_feature_context else None  # once per country
    feat_kw = {} if ctx is None else {"ctx": ctx}
    for start in range(0, len(s1), chunk_s1):
        chunk = s1.iloc[start:start + chunk_s1]
        cands = comps.generate_candidates(chunk, index)
        if cand_path is not None:
            append_id_lists(cands, chunk["entity_id"], cand_path)
        if cands.empty:
            continue
        yield cands, comps.build_features(cands, chunk, pool, **feat_kw)


def _score(cands: pd.DataFrame, feats: pd.DataFrame, model) -> pd.DataFrame:
    proba = model.predict_proba(feats)
    keep = proba >= PROBA_FLOOR
    return cands.loc[keep, [S1_ID, CAND_ID]].assign(**{PROBA: proba[keep]}).reset_index(drop=True)


def _concat(parts, empty: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(parts, ignore_index=True) if parts else empty


def score_country(s1: pd.DataFrame, pool: pd.DataFrame, model, cand_path: Path | None,
                  comps: Components = REAL, chunk_s1: int = CHUNK_S1) -> pd.DataFrame:
    """Block, featurize and score every S1 chunk of one country. Returns (s1_id, cand_id, proba), proba >= floor."""
    parts = []
    for cands, feats in _country_chunks(s1, pool, comps, chunk_s1, cand_path):
        parts.append(_score(cands, feats, model))
        del cands, feats
        gc.collect()
    return _concat(parts, _NO_PAIRS.assign(**{PROBA: []}))


# ------------------------------------------------------------------ test
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


# ------------------------------------------------------------------ train / dev (dev world)
def load_world(country: str | None = None, limit_s1: int | None = None) -> DevWorld:
    """Frozen dev world, optionally one country and/or the first `limit_s1` S1 (by id) per country."""
    w = load_dev_world(country)
    if limit_s1:
        s1 = w.s1.sort_index().groupby("country", sort=True, group_keys=False).head(limit_s1)
        w = DevWorld(s1, w.s2, w.s3, w.truth_raw[w.truth_raw["source1_entity_id"].isin(s1.index)])
    return w


def _world_countries(w: DevWorld) -> list[str]:
    return sorted(w.s1["country"].unique())  # from the data, never hard-coded


def featurize_world(w: DevWorld, s1_ids, comps: Components, chunk_s1: int = CHUNK_S1, log: Callable = print):
    """Run the given dev-world S1 through the shared chunk loop. Returns (pairs, feats), aligned RangeIndex."""
    wanted = set(s1_ids)
    pool_all = w.pool
    pair_parts, feat_parts = [], []
    for country in _world_countries(w):
        t = time.perf_counter()
        s1_raw = w.s1[(w.s1["country"] == country) & w.s1.index.isin(wanted)]
        if s1_raw.empty:
            continue
        s1 = comps.normalize_records(s1_raw)
        pool = comps.normalize_records(pool_all[pool_all["country"] == country])
        n = 0
        for cands, feats in _country_chunks(s1, pool, comps, chunk_s1):
            pair_parts.append(cands[[S1_ID, CAND_ID]].reset_index(drop=True))
            feat_parts.append(feats.reset_index(drop=True))
            n += len(cands)
        log(f"{country}: {len(s1):,} S1, pool {len(pool):,} -> {n:,} pairs ({time.perf_counter() - t:.0f}s)")
        del s1, pool
        gc.collect()
    pairs = _concat(pair_parts, _NO_PAIRS.copy())
    feats = _concat(feat_parts, pd.DataFrame())
    return pairs, feats


def choose_threshold(sweep: pd.DataFrame) -> float:
    """CONTRACT §7: highest validation macro_f05; ties -> highest threshold (precision first)."""
    top = sweep["macro_f05"].max()
    return float(sweep.loc[sweep["macro_f05"] >= top - 1e-12, "threshold"].max())


def _md(title: str, df: pd.DataFrame) -> str:
    return f"## {title}\n\n```\n{df.to_string(index=False, float_format=lambda v: f'{v:.4f}')}\n```\n"


def run_train(w: DevWorld, comps: Components = REAL, scoring: Scoring = REAL_SCORING, chunk_s1: int = CHUNK_S1,
              one_to_one: bool = ONE_TO_ONE, model_path: Path = MODEL_PATH,
              report_path: Path = OUTPUT_DIR / "train_report.md", log: Callable = print):
    """Train on the TRAIN split, tune the threshold on the VAL split, save model + threshold. Returns the model."""
    if comps.train_model is None:
        raise ValueError("Components.train_model is required for `predict train`")
    truth = evaluate.truth_dict(w.truth_raw)
    train_ids = sorted(w.s1.index[w.s1["split"] == "train"])
    val_ids = sorted(w.s1.index[w.s1["split"] == "val"])
    fit_ids, es_ids = scoring.split_s1(train_ids, val_frac=ES_FRAC)
    assert not set(fit_ids) & set(val_ids) and not set(es_ids) & set(val_ids), "validation S1 leaked into fitting"

    pairs, feats = featurize_world(w, w.s1.index, comps, chunk_s1, log)
    labels = scoring.label_pairs(pairs, truth)
    owner = pairs[S1_ID]
    m_fit, m_es, m_val = owner.isin(set(fit_ids)), owner.isin(set(es_ids)), owner.isin(set(val_ids))
    log(f"{len(pairs):,} pairs ({int(labels.sum()):,} positive); fit {int(m_fit.sum()):,} / "
        f"early-stop {int(m_es.sum()):,} / validation {int(m_val.sum()):,}")

    es = (feats[m_es], labels[m_es]) if m_es.any() else (None, None)
    model = comps.train_model(feats[m_fit], labels[m_fit], *es)
    scored_val = _score(pairs[m_val], feats[m_val], model)

    def select(sc, t):
        return comps.select_matches(sc, t, one_to_one=one_to_one)

    sweep = scoring.sweep_thresholds(scored_val, truth, val_ids, select, THRESHOLD_GRID)
    model.threshold = choose_threshold(sweep)
    model.save(model_path)

    best = sweep.loc[sweep["threshold"] == model.threshold].iloc[0]
    s1_country = w.s1["country"]
    val_truth = {s: truth.get(s, set()) for s in val_ids}
    report = [
        "# predict train report", "",
        f"- Data: frozen dev world; {len(fit_ids):,} fit / {len(es_ids):,} early-stop / {len(val_ids):,} validation S1; "
        f"one_to_one={one_to_one}",
        f"- **Chosen threshold {model.threshold:.2f} -> validation macro F0.5 {best['macro_f05']:.4f}** "
        f"(precision {best['precision']:.4f}, recall {best['recall']:.4f})", "",
        _md("Blocking (all dev S1)", scoring.blocking_report(pairs, truth, s1_country)),
        _md("Threshold sweep (validation)", sweep),
        _md("Per country at the chosen threshold (validation)",
            scoring.per_country_report(select(scored_val, model.threshold), val_truth, s1_country.loc[val_ids])),
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    log(f"threshold {model.threshold:.2f}, validation macro F0.5 {best['macro_f05']:.4f}; "
        f"model -> {model_path}, report -> {report_path}")
    return model


def run_dev(w: DevWorld, model, comps: Components = REAL, scoring: Scoring = REAL_SCORING, chunk_s1: int = CHUNK_S1,
            one_to_one: bool = ONE_TO_ONE, report_path: Path = OUTPUT_DIR / "dev_report.md",
            log: Callable = print) -> float:
    """Score the VAL split with a saved model at model.threshold. Returns validation macro F0.5."""
    truth = evaluate.truth_dict(w.truth_raw)
    val_ids = sorted(w.s1.index[w.s1["split"] == "val"])
    pairs, feats = featurize_world(w, val_ids, comps, chunk_s1, log)
    scored = _score(pairs, feats, model) if len(pairs) else _NO_PAIRS.assign(**{PROBA: []})
    matches = comps.select_matches(scored, model.threshold, one_to_one=one_to_one)
    val_truth = {s: truth.get(s, set()) for s in val_ids}
    pred = matches.groupby(S1_ID)[CAND_ID].agg(set).to_dict() if len(matches) else {}
    f05 = evaluate.macro_f_beta(pred, val_truth)
    s1_country = w.s1["country"]
    report = [
        "# predict dev report", "",
        f"- {len(val_ids):,} validation S1; threshold {model.threshold:.2f}; one_to_one={one_to_one}",
        f"- **Validation macro F0.5 {f05:.4f}**", "",
        _md("Blocking (validation S1)", scoring.blocking_report(pairs, val_truth, s1_country.loc[val_ids])),
        _md("Per country", scoring.per_country_report(matches, val_truth, s1_country.loc[val_ids])),
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    log(f"validation macro F0.5 {f05:.4f} at threshold {model.threshold:.2f}; report -> {report_path}")
    return f05


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["train", "dev", "test"])
    ap.add_argument("--dry-run", action="store_true", help="test mode: write an empty but valid submission")
    ap.add_argument("--country", help="train/dev: restrict to one country")
    ap.add_argument("--limit-s1", type=int, help="train/dev: first N S1 (by id) per country, for smoke runs")
    args = ap.parse_args()
    if args.mode == "test":
        run_test(dry_run=args.dry_run)
    elif args.mode == "train":
        run_train(load_world(args.country, args.limit_s1))
    else:
        run_dev(load_world(args.country, args.limit_s1), matcher.Model.load(MODEL_PATH))


if __name__ == "__main__":
    main()
