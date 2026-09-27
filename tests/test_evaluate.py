"""Person 4 tests: evaluate.py (split, labels, contract checks, blocking report, validation scoring) and
scripts/package_submission.py (submission zip).

    python -m unittest discover -s tests -t . -v

Scoring is cross-checked against Person 3's notebooks/train_matcher.score / sweep (the code behind the
0.9400 at threshold 0.70 in PROGRESS.md) on random scored frames, with matcher.select_matches injected,
one-to-one off and on. On real data run `python -m notebooks.check_evaluate` (needs the dev world).
"""
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

import tests.test_contract as tc
from scripts import package_submission as ps
from src import evaluate as ev
from src.config import BLK_COLUMNS, CAND_ID, PROBA, S1_ID, SEED
from src.predict import run_pipeline


def toy_country(country="France"):
    (s1, s2, s3), truth = tc.build_toy()
    idx = lambda d: d[d["country"] == country].set_index("entity_id", drop=False)  # noqa: E731
    s1, pool = tc.fake_normalize(idx(s1)), tc.fake_normalize(pd.concat([idx(s2), idx(s3)]))
    pairs = tc.fake_generate_candidates(s1, tc.fake_build_index(pool))
    return s1, pool, pairs, truth


def random_world(n_s1=400, seed=SEED):
    """Random truth + scored pairs: singletons, S1 with no scored rows, shared candidates, ties in proba."""
    rng = np.random.default_rng(seed)
    ids = [f"S1-{i:05d}" for i in range(n_s1)]
    country = pd.Series(rng.choice(["US", "India", "Xland"], n_s1), index=ids)
    truth, rows, nxt = {}, [], 0
    for s in ids:
        k = int(rng.choice([0, 1, 2, 3, 5], p=[0.1, 0.2, 0.3, 0.3, 0.1]))
        truth[s] = {f"S{rng.integers(2, 4)}-{nxt + j}" for j in range(k)}
        nxt += k
        if rng.random() < 0.1:
            continue  # no candidates at all
        for c in truth[s]:
            if rng.random() < 0.9:
                rows.append((s, c, rng.beta(5, 2)))
        for _ in range(int(rng.integers(0, 6))):  # decoys, some shared with other S1
            rows.append((s, f"S{rng.integers(2, 4)}-{int(rng.integers(0, nxt + 50))}", rng.beta(2, 5)))
    scored = pd.DataFrame(rows, columns=[S1_ID, CAND_ID, PROBA]).drop_duplicates([S1_ID, CAND_ID])
    scored[PROBA] = (np.round(scored[PROBA] * 20) / 20).astype(np.float32)  # ties on purpose
    return ids, country, truth, scored.reset_index(drop=True)


# ------------------------------------------------------------------ split_s1
class SplitTest(unittest.TestCase):
    IDS = [f"S1-{i}" for i in range(40)]

    def test_golden_values_same_on_every_machine(self):
        tr, va = ev.split_s1(self.IDS, 0.25)
        self.assertEqual(va, ["S1-1", "S1-4", "S1-9", "S1-12", "S1-17", "S1-18", "S1-25", "S1-27", "S1-28", "S1-34"])
        self.assertEqual(ev.split_s1(self.IDS, 0.25, seed=7)[1],
                         ["S1-3", "S1-5", "S1-9", "S1-11", "S1-14", "S1-18", "S1-19", "S1-27", "S1-28", "S1-34",
                          "S1-36"])

    def test_partition_order_and_row_order_independence(self):
        ids = [f"S1-{i}" for i in range(20_000)]
        tr, va = ev.split_s1(ids, 0.1)
        self.assertFalse(set(tr) & set(va))
        self.assertEqual(sorted(tr + va), sorted(ids))
        self.assertEqual(tr, [s for s in ids if s in set(tr)], "input order kept")
        self.assertAlmostEqual(len(va) / len(ids), 0.1, delta=0.01)
        tr2, va2 = ev.split_s1(ids[::-1], 0.1)
        self.assertEqual(set(va2), set(va), "membership depends on the id only")
        self.assertEqual(set(ev.split_s1(ids[:500], 0.1)[1]), set(va) & set(ids[:500]), "not on the other ids")
        self.assertLessEqual(set(ev.split_s1(ids, 0.05)[1]), set(va), "smaller val_frac -> subset")
        self.assertNotEqual(set(ev.split_s1(ids, 0.1, seed=1)[1]), set(va), "seed matters")

    def test_accepts_index_series_and_arrow_strings(self):
        ids = pd.Index(self.IDS, dtype=pd.ArrowDtype(pa.string()))
        want = ev.split_s1(self.IDS, 0.25)
        self.assertEqual(ev.split_s1(ids, 0.25), want)
        self.assertEqual(ev.split_s1(pd.Series(self.IDS), 0.25), want)
        self.assertEqual(ev.split_s1([], 0.25), ([], []))

    def test_frozen_dev_world_split_column_is_used_as_is(self):
        s1 = pd.DataFrame({"split": ["val", "train", "train", "val"]}, index=["S1-1", "S1-2", "S1-3", "S1-4"])
        self.assertEqual(ev.split_s1(s1, val_frac=0.9), (["S1-2", "S1-3"], ["S1-1", "S1-4"]))
        with self.assertRaises(ValueError):
            ev.split_s1(s1.rename(columns={"split": "x"}))
        with self.assertRaises(ValueError):
            ev.split_s1(s1.assign(split=["val", "test", "train", "val"]))

    def test_hash_is_the_dev_world_recipe(self):
        from scripts.build_dev_world import SALT, u01
        ids = [f"S1-{i}" for i in range(0, 5_000_000, 997)]
        for salt in SALT.values():
            np.testing.assert_array_equal(ev._u01(ids, salt), u01(pa.chunked_array([ids]), salt, SEED))

    def test_early_stopping_carve_out_of_dev_train_split_is_not_empty(self):
        # predict train does split_s1(train_ids, val_frac=ES_FRAC); reusing the frozen salt would give nothing
        from scripts.build_dev_world import SALT
        ids = [f"S1-{i}" for i in range(20_000)]
        train_ids = [s for s, u in zip(ids, ev._u01(ids, SALT["split"])) if u >= 0.2]
        fit, es = ev.split_s1(train_ids, val_frac=0.15)
        self.assertAlmostEqual(len(es) / len(train_ids), 0.15, delta=0.01)

    def test_rejects_ids_without_numeric_part(self):
        with self.assertRaisesRegex(ValueError, "S1-<digits>"):
            ev.split_s1(["S1-abc"])


# ------------------------------------------------------------------ label_pairs
class LabelPairsTest(unittest.TestCase):
    def test_values_dtype_and_alignment(self):
        pairs = pd.DataFrame({S1_ID: ["S1-1", "S1-1", "S1-2", "S1-9"], CAND_ID: ["S2-1", "S3-4", "S2-1", "S2-5"]},
                             index=[10, 3, 7, 0])
        truth = {"S1-1": {"S2-1"}, "S1-2": set()}
        y = ev.label_pairs(pairs, truth)
        self.assertEqual(y.dtype, np.int8)
        self.assertTrue(y.index.equals(pairs.index))
        self.assertEqual(y.tolist(), [1, 0, 0, 0], "S1-2 is a singleton, S1-9 has no truth")

    def test_arrow_strings_and_empty(self):
        arrow = pd.ArrowDtype(pa.string())
        pairs = pd.DataFrame({S1_ID: pd.Series(["S1-1"], dtype=arrow), CAND_ID: pd.Series(["S2-1"], dtype=arrow)})
        self.assertEqual(ev.label_pairs(pairs, {"S1-1": {"S2-1"}}).tolist(), [1])
        empty = ev.label_pairs(pairs.iloc[:0], {"S1-1": {"S2-1"}})
        self.assertEqual((len(empty), empty.dtype), (0, np.int8))

    def test_matches_person3_labelling(self):
        _, _, truth, scored = random_world()
        ref = np.fromiter((c in truth.get(s, ()) for s, c in zip(scored[S1_ID], scored[CAND_ID])), np.int8,
                          len(scored))  # notebooks/train_matcher.py
        np.testing.assert_array_equal(ev.label_pairs(scored, truth).to_numpy(), ref)


# ------------------------------------------------------------------ check_pairs / check_feats
class CheckPairsTest(unittest.TestCase):
    def setUp(self):
        self.s1, self.pool, self.pairs, _ = toy_country()

    def assertFails(self, pairs, msg, **kw):
        with self.assertRaisesRegex(AssertionError, msg):
            ev.check_pairs(pairs, self.s1, self.pool, **kw)

    def test_valid_pairs_pass(self):
        ev.check_pairs(self.pairs, self.s1, self.pool)
        ev.check_pairs(self.pairs.iloc[:0], self.s1, self.pool)

    def test_each_invariant(self):
        p = self.pairs
        s1_as_cand = p.assign(**{CAND_ID: p[CAND_ID].where(p.index != 0, self.s1.index[0])})
        self.assertFails(s1_as_cand, "not S2-/S3-")
        self.assertFails(pd.concat([p, p.iloc[[0]]], ignore_index=True), "duplicate \\(s1_id, cand_id\\)")
        self.assertFails(p, "more than 1 candidates", max_cands=1)
        self.assertFails(p.assign(**{S1_ID: p[S1_ID].where(p.index != 0, "S1-999")}), "not in the given S1 chunk")
        self.assertFails(p.assign(**{CAND_ID: p[CAND_ID].where(p.index != 0, "S2-999")}), "not in the pool")
        self.assertFails(p.drop(columns="blk_addr_rank"), "missing columns \\['blk_addr_rank'\\]")
        self.assertFails(p.set_axis([0] * len(p)), "duplicate labels")
        self.assertFails(p.assign(blk_name_score=np.inf), "blk_name_score contains inf")
        self.assertFails(p.assign(blk_name_rank="1"), "blk_name_rank is not numeric")

    def test_cap_boundary(self):
        top = int(self.pairs[S1_ID].value_counts().max())
        ev.check_pairs(self.pairs, self.s1, self.pool, max_cands=top)
        self.assertFails(self.pairs, f"more than {top - 1} candidates", max_cands=top - 1)

    def test_cross_country_pair(self):
        _, us_pool, _, _ = toy_country("US")
        pool = pd.concat([self.pool, us_pool])
        bad = pd.concat([self.pairs, pd.DataFrame({S1_ID: [self.s1.index[0]], CAND_ID: [us_pool.index[0]],
                                                   **{c: [0.0] for c in BLK_COLUMNS}})], ignore_index=True)
        with self.assertRaisesRegex(AssertionError, "1 pairs across countries"):
            ev.check_pairs(bad, self.s1, pool)

    def test_reports_every_problem_at_once(self):
        bad = pd.concat([self.pairs, self.pairs.iloc[[0]]], ignore_index=True).assign(blk_name_score=np.inf)
        with self.assertRaises(AssertionError) as cm:
            ev.check_pairs(bad, self.s1, self.pool)
        self.assertIn("duplicate", str(cm.exception))
        self.assertIn("inf", str(cm.exception))


class CheckFeatsTest(unittest.TestCase):
    def setUp(self):
        self.s1, self.pool, self.pairs, _ = toy_country()
        self.feats = tc.fake_build_features(self.pairs, self.s1, self.pool)
        self.names = list(self.feats.columns)

    def assertFails(self, feats, msg, pairs=None, names=None):
        with self.assertRaisesRegex(AssertionError, msg):
            ev.check_feats(feats, self.pairs if pairs is None else pairs, names)

    def test_valid_feats_pass(self):
        ev.check_feats(self.feats, self.pairs, self.names)
        ev.check_feats(self.feats.assign(name_jaccard=np.float32("nan")), self.pairs)  # NaN allowed

    def test_each_invariant(self):
        f = self.feats
        self.assertFails(f.iloc[::-1], "feats.index != pairs.index")
        self.assertFails(f.iloc[1:], "feature rows")
        self.assertFails(f.set_axis(f.index + 100), "feats.index != pairs.index")
        self.assertFails(f.astype("float64"), "expected float32")
        self.assertFails(f.assign(addr_equal=np.float32("inf")), "addr_equal has .* inf")
        self.assertFails(f, "missing \\['x'\\]", names=self.names + ["x"])
        self.assertFails(f[self.names[::-1]], "wrong order", names=self.names)
        self.assertFails(pd.concat([f, f[["name_jaccard"]]], axis=1), "duplicate columns")

    def test_real_features_module_passes(self):
        from src.features import FEATURE_NAMES, build_features
        feats = build_features(self.pairs, self.s1, self.pool)
        ev.check_feats(feats, self.pairs, FEATURE_NAMES)


# ------------------------------------------------------------------ blocking_report
class BlockingReportTest(unittest.TestCase):
    def setUp(self):
        self.truth = {"S1-1": {"S2-1", "S3-1"}, "S1-2": {"S2-2"}, "S1-3": set(), "S1-4": {"S3-5"}}
        self.country = pd.Series({"S1-1": "US", "S1-2": "US", "S1-3": "India", "S1-4": "India"})
        self.pairs = pd.DataFrame(
            [("S1-1", "S2-1"), ("S1-1", "S2-9"), ("S1-2", "S3-7"), ("S1-4", "S3-5"), ("S1-4", "S2-8"),
             ("S1-1", "S2-1"),     # duplicate: counted once
             ("S1-99", "S2-2")],   # S1 outside the universe: ignored
            columns=[S1_ID, CAND_ID])
        self.rep = ev.blocking_report(self.pairs, self.truth, self.country).set_index(["country", "source"])

    def row(self, c, s):
        return self.rep.loc[(c, s)]

    def test_hand_computed_numbers(self):
        us = self.row("US", "all")
        self.assertEqual((us["n_s1"], us["true_pairs"], us["found_pairs"]), (2, 3, 1))
        self.assertAlmostEqual(us["recall"], 1 / 3)
        self.assertAlmostEqual(us["avg_cands_per_s1"], 1.5)
        self.assertEqual((us["max_cands_per_s1"], us["s1_no_cands"], us["s1_all_found"]), (2, 0.0, 0.0))
        us2, us3 = self.row("US", "S2"), self.row("US", "S3")
        self.assertEqual((us2["true_pairs"], us2["found_pairs"], us2["recall"], us2["avg_cands_per_s1"],
                          us2["s1_no_cands"]), (2, 1, 0.5, 1.0, 0.5))
        self.assertEqual((us3["true_pairs"], us3["found_pairs"], us3["recall"], us3["avg_cands_per_s1"]),
                         (1, 0, 0.0, 0.5))
        india = self.row("India", "all")
        self.assertEqual((india["n_s1"], india["recall"], india["s1_all_found"], india["s1_no_cands"]),
                         (2, 1.0, 1.0, 0.5))
        self.assertTrue(np.isnan(self.row("India", "S2")["recall"]), "no true S2 pairs -> NaN, not 1.0")
        al = self.row("ALL", "all")
        self.assertEqual((al["n_s1"], al["true_pairs"], al["found_pairs"], al["avg_cands_per_s1"]),
                         (4, 4, 2, 1.25))

    def test_layout(self):
        self.assertEqual(list(self.rep.index.get_level_values(0).unique()), ["India", "US", "ALL"])
        self.assertEqual(list(self.rep.loc["ALL"].index), ["all", "S2", "S3"])

    def test_overall_recall_equals_candidate_recall(self):
        ids, country, truth, scored = random_world()
        rep = ev.blocking_report(scored, truth, country).set_index(["country", "source"])
        cands = scored.groupby(S1_ID)[CAND_ID].agg(set).to_dict()
        self.assertAlmostEqual(rep.loc[("ALL", "all"), "recall"], ev.candidate_recall(cands, truth), places=12)
        by_src = rep.loc[("ALL", "S2"), "found_pairs"] + rep.loc[("ALL", "S3"), "found_pairs"]
        self.assertEqual(by_src, rep.loc[("ALL", "all"), "found_pairs"])

    def test_empty_pairs(self):
        rep = ev.blocking_report(self.pairs.iloc[:0], self.truth, self.country).set_index(["country", "source"])
        self.assertEqual(rep.loc[("ALL", "all"), "recall"], 0.0)
        self.assertEqual(rep.loc[("ALL", "all"), "s1_no_cands"], 1.0)


# ------------------------------------------------------------------ sweep / per country / LOCO
class ScoringTest(unittest.TestCase):
    GRID = [round(0.10 + 0.05 * i, 2) for i in range(18)]

    @classmethod
    def setUpClass(cls):
        cls.ids, cls.country, cls.truth, cls.scored = random_world()

    def test_sweep_reproduces_person3_train_matcher_exactly(self):
        import notebooks.train_matcher as tm
        from src.matcher import select_matches
        for o2o in (False, True):
            ours = ev.sweep_thresholds(self.scored, self.truth, self.ids,
                                       lambda sc, t: select_matches(sc, t, one_to_one=o2o), tm.GRID)
            ref = tm.sweep(self.scored, self.truth, self.ids, one_to_one=o2o)
            for col in ("threshold", "precision", "recall", "macro_f05", "singleton_acc", "pred_pairs"):
                np.testing.assert_allclose(ours[col].to_numpy(float), ref[col].to_numpy(float), rtol=0,
                                           atol=1e-12, err_msg=f"{col}, one_to_one={o2o}")
            self.assertEqual(ev._best_row(ours)["threshold"], tm.best(ref)["threshold"])

    def test_sweep_matches_official_metric_and_counts_every_s1(self):
        sel = tc.fake_select_matches
        sweep = ev.sweep_thresholds(self.scored, self.truth, self.ids, sel, self.GRID)
        self.assertEqual(sweep["threshold"].tolist(), self.GRID)
        self.assertTrue({"threshold", "precision", "recall", "macro_f05"} <= set(sweep.columns))
        for t, f in zip(sweep["threshold"], sweep["macro_f05"]):
            m = sel(self.scored, t)
            pred = m.groupby(S1_ID)[CAND_ID].agg(set).to_dict()
            self.assertAlmostEqual(f, ev.macro_f_beta(pred, {s: self.truth[s] for s in self.ids}), places=12)
        self.assertTrue((sweep["n_s1"] == len(self.ids)).all())

    def test_edge_cases(self):
        truth = {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S3-1", "S2-3"}}
        perfect = pd.DataFrame({S1_ID: ["S1-1", "S1-3", "S1-3"], CAND_ID: ["S2-1", "S3-1", "S2-3"], PROBA: 0.9})
        sw = ev.sweep_thresholds(perfect, truth, list(truth), tc.fake_select_matches, [0.5, 0.95])
        self.assertEqual(sw["macro_f05"].tolist(), [1.0, 1 / 3], "nothing predicted: only the singleton scores")
        self.assertEqual(sw.loc[1, "precision"], 1.0, "no predictions -> precision 1.0 (as train_matcher)")
        # a prediction for an S1 outside s1_ids is ignored; a wrong one on the singleton costs it
        extra = pd.concat([perfect, pd.DataFrame({S1_ID: ["S1-7", "S1-2"], CAND_ID: ["S2-9", "S2-8"], PROBA: 0.9})])
        self.assertAlmostEqual(ev.sweep_thresholds(extra, truth, list(truth), tc.fake_select_matches,
                                                   [0.5])["macro_f05"][0], 2 / 3)
        self.assertEqual(ev.sweep_thresholds(perfect, truth, [], tc.fake_select_matches, [0.5])["macro_f05"][0], 0.0)
        # a select_fn returning the same pair twice must not inflate precision or pred_pairs
        doubled = ev.sweep_thresholds(extra, truth, list(truth), lambda sc, t: pd.concat([sc, sc]), [0.5])
        once = ev.sweep_thresholds(extra, truth, list(truth), lambda sc, t: sc, [0.5])
        pd.testing.assert_frame_equal(doubled, once)
        self.assertEqual(once.loc[0, "pred_pairs"], 4)

    def test_per_country_report(self):
        matches = tc.fake_select_matches(self.scored, 0.5)
        rep = ev.per_country_report(matches, self.truth, self.country).set_index("country")
        self.assertEqual(list(rep.index), ["India", "US", "Xland", "ALL"])
        pred = matches.groupby(S1_ID)[CAND_ID].agg(set).to_dict()
        for c in ("India", "US", "Xland"):
            ids = self.country.index[self.country == c]
            self.assertEqual(rep.loc[c, "n_s1"], len(ids))
            self.assertAlmostEqual(rep.loc[c, "macro_f05"], ev.macro_f_beta(pred, {s: self.truth[s] for s in ids}),
                                   places=12)
        weighted = (rep.loc[["India", "US", "Xland"], "macro_f05"] * rep.loc[["India", "US", "Xland"], "n_s1"]).sum()
        self.assertAlmostEqual(rep.loc["ALL", "macro_f05"], weighted / len(self.ids), places=12)

    def test_loco_report(self):
        calls = []

        def score_held_out(c):
            calls.append(c)
            return self.scored[self.scored[S1_ID].map(self.country) == c]

        rep = ev.loco_report(score_held_out, self.truth, self.country, tc.fake_select_matches, self.GRID, 0.5)
        self.assertEqual(calls, ["India", "US", "Xland"])
        self.assertEqual(rep["train_on"].tolist(), ["US+Xland", "India+Xland", "India+US"])
        per = ev.per_country_report(tc.fake_select_matches(self.scored, 0.5), self.truth,
                                    self.country).set_index("country")
        for _, r in rep.iterrows():
            self.assertAlmostEqual(r["f05_at_threshold"], per.loc[r["validate_on"], "macro_f05"], places=12)
            self.assertGreaterEqual(r["best_f05_there"], r["f05_at_threshold"] - 1e-12)

    def test_threshold_rule_ties_go_to_highest(self):
        sweep = pd.DataFrame({"threshold": [0.3, 0.5, 0.7, 0.9], "macro_f05": [0.8, 0.9, 0.9, 0.85]})
        self.assertEqual(ev._best_row(sweep)["threshold"], 0.7)


# ------------------------------------------------------------------ submission package
FAKE_VALIDATOR = """import sys
ok = open(sys.argv[sys.argv.index('-m') + 1], encoding='utf-8').read().startswith('source1_entity_id')
print('PASS' if ok else 'FAIL: bad header'); sys.exit(0 if ok else 1)
"""


class PackageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        r = self.root = Path(self.tmp.name)
        (s1, s2, s3), _ = tc.build_toy()
        self.test_dir = r / "test"
        self.test_dir.mkdir()
        for n, df in (("1", s1), ("2", s2), ("3", s3)):
            df.to_csv(self.test_dir / f"test_source{n}.tsv", sep="\t", index=False, lineterminator="\n")

        def load_country(c):
            idx = lambda d: d[d["country"] == c].set_index("entity_id", drop=False)  # noqa: E731
            return idx(s1), pd.concat([idx(s2), idx(s3)])

        self.out = r / "outputs"
        run_pipeline(tc.COUNTRIES, load_country, tc.FakeModel(), self.out, tc.FAKES, 4, False, log=lambda *_: None)
        self.repo = r / "repo"
        for f, text in (("src/a.py", "x = 1\n"), ("src/sub/b.py", "y = 2\n"), ("src/__pycache__/c.py", "junk\n"),
                        ("scripts/s.py", "z = 3\n"), ("requirements.txt", "pandas==3.0.6\n# comment\n")):
            (self.repo / f).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / f).write_text(text, encoding="utf-8")
        self.doc, self.readme, self.validator = r / "doc.md", r / "readme.md", r / "validate.py"
        self.doc.write_text("# Method\n", encoding="utf-8")
        self.readme.write_text("# Reproduce\n", encoding="utf-8")
        self.validator.write_text(FAKE_VALIDATOR, encoding="utf-8")
        self.zip = r / "sub.zip"

    def tearDown(self):
        self.tmp.cleanup()

    def package(self, **kw):
        args = dict(matches=self.out / "matching_results.tsv", candidates=self.out / "candidate_pairs.tsv",
                    doc=self.doc, readme=self.readme, out=self.zip, test_dir=self.test_dir,
                    validator=self.validator, repo=self.repo, log=lambda *_: None)
        return ps.package(**{**args, **kw})

    def assertFails(self, msg, **kw):
        with self.assertRaisesRegex(ps.PackagingError, msg):
            self.package(**kw)
        self.assertFalse(self.zip.exists(), "a failed run must leave no zip")
        self.assertFalse(list(self.root.glob("*.partial")))

    def write(self, name, lines):
        path = self.root / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        return path

    def test_zip_layout_and_contents(self):
        self.package()
        with zipfile.ZipFile(self.zip) as z:
            code = "code/business_entity_resolution/"
            self.assertEqual(sorted(z.namelist()), sorted([
                "Documentation_template.md", "output/candidate_pairs.tsv", "output/matching_results.tsv",
                code + "README.md", code + "requirements.txt", code + "scripts/s.py", code + "src/a.py",
                code + "src/sub/b.py"]))
            self.assertEqual(z.read("output/matching_results.tsv"), (self.out / "matching_results.tsv").read_bytes())
        first = self.zip.read_bytes()
        self.package()
        self.assertEqual(self.zip.read_bytes(), first, "same inputs -> byte-identical zip")

    def test_output_problems_fail_before_the_validator(self):
        m = (self.out / "matching_results.tsv").read_text(encoding="utf-8").splitlines()
        head, rows = m[0], m[1:]
        s1 = rows[0].split("\t")[0]
        self.assertFails("not a subset", matches=self.write("m1.tsv", [head, f"{s1}\tS2-999", *rows[1:]]))
        self.assertFails("header", matches=self.write("m2.tsv", ["a\tb", *rows]))
        self.assertFails("different number of rows", matches=self.write("m3.tsv", [head, *rows[:-1]]))
        self.assertFails("S1 order differs", matches=self.write("m4.tsv", [head, *rows[::-1]]))
        self.assertFails("non S2-/S3- id", matches=self.write("m5.tsv", [head, f"{s1}\t{s1}", *rows[1:]]))

    def test_row_count_must_equal_test_s1(self):
        f = self.test_dir / "test_source1.tsv"
        f.write_text(f.read_text(encoding="utf-8") + "S1-999\tX\tY\tUS\n", encoding="utf-8")
        self.assertFails("24 S1 rows but 25")

    def test_validator_failure_is_loud(self):
        self.validator.write_text("import sys; print('ERROR: ids missing'); sys.exit(1)\n", encoding="utf-8")
        self.assertFails("(?s)official validator failed .*ids missing")
        self.validator.write_text("print('looks fine')\n", encoding="utf-8")  # exit 0 but no PASS
        self.assertFails("official validator failed")

    def test_placeholders_unpinned_requirements_and_network_code(self):
        self.doc.write_text("Score: [[TBD]]\n", encoding="utf-8")
        self.assertFails("placeholder")
        self.package(allow_placeholders=True)
        self.zip.unlink()
        self.doc.write_text("# ok\n", encoding="utf-8")
        (self.repo / "requirements.txt").write_text("pandas>=2\n", encoding="utf-8")
        self.assertFails("not pinned")
        (self.repo / "requirements.txt").write_text("pandas==3.0.6\n", encoding="utf-8")
        (self.repo / "src" / "net.py").write_text("import requests\n", encoding="utf-8")
        self.assertFails("network access")
        (self.repo / "src" / "net.py").write_text("U = 'ht' 'tp:' '//x'\nV = 'https://example.org'\n", encoding="utf-8")
        self.assertFails("network access / URL 'https://'")

    def test_stale_zip_is_removed_when_a_run_fails(self):
        self.package()
        self.validator.write_text("import sys; sys.exit(1)\n", encoding="utf-8")
        self.assertFails("validator failed")

    def test_missing_inputs_listed(self):
        self.assertFails("official validator", validator=self.root / "nope.py")

    def test_real_repo_code_passes_review(self):
        ps.check_code(ps.ROOT, ps.ROOT / "requirements.txt")

    def test_official_validator_on_toy_output(self):
        if not ps.VALIDATOR.exists():
            self.skipTest(f"official validator not found at {ps.VALIDATOR} (unzip student_resource into data/raw/)")
        self.package(validator=ps.VALIDATOR)
        self.assertTrue(self.zip.exists())


if __name__ == "__main__":
    unittest.main()
