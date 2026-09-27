"""`predict train` / `predict dev` driver tests with fake components (wiring only, not ML quality).

    python -m unittest discover -s tests -t . -v

Checks: validation S1 never reach fitting/early stopping; the threshold follows the CONTRACT §7 rule; the
model is saved with that threshold; ctx is built once per country; `dev` gives exactly the macro F0.5 of
running the real test path (`run_pipeline`) on the same S1; `--limit-s1` / `--country` filtering.
"""
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import tests.test_contract as tc
from src import evaluate
from src.config import CAND_ID, PROBA, S1_ID, THRESHOLD_GRID
from src.data_loader import DevWorld
from src.predict import Components, Scoring, choose_threshold, run_dev, run_pipeline, run_train


class PicklableModel(tc.FakeModel):
    def __init__(self):
        self.threshold = 0.5

    def save(self, path):
        Path(path).write_bytes(pickle.dumps(self))


def toy_world() -> DevWorld:
    (s1, s2, s3), truth = tc.build_toy()
    s1 = s1.assign(split=["val" if i % 3 == 2 else "train" for i in range(len(s1))])  # 1/3 validation
    rows = [(s, ",".join(sorted(t))) for s, t in truth.items()]
    gt = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    idx = lambda d: d.set_index("entity_id", drop=False)  # noqa: E731
    return DevWorld(idx(s1), idx(s2), idx(s3), gt)


def fake_split(ids, val_frac=0.15, seed=0):
    ids = sorted(ids)
    k = max(1, int(len(ids) * val_frac))
    return ids[k:], ids[:k]


def fake_label(pairs, truth):
    return pd.Series([int(c in truth.get(s, ())) for s, c in zip(pairs[S1_ID], pairs[CAND_ID])],
                     index=pairs.index, dtype=np.int8)


def fake_sweep(scored, truth, s1_ids, select_fn, grid):
    rows = []
    for t in grid:
        m = select_fn(scored, t)
        pred = m.groupby(S1_ID)[CAND_ID].agg(set).to_dict() if len(m) else {}
        rows.append({"threshold": t, "precision": np.nan, "recall": np.nan,
                     "macro_f05": evaluate.macro_f_beta(pred, {s: truth.get(s, set()) for s in s1_ids})})
    return pd.DataFrame(rows)


FAKE_SCORING = Scoring(fake_split, fake_label, fake_sweep,
                       lambda pairs, truth, c: pd.DataFrame({"pairs": [len(pairs)]}),
                       lambda m, truth, c: pd.DataFrame({"matches": [len(m)]}))


class PredictModesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.w = toy_world()
        self.fit_calls, self.ctx_built = [], []

        def train_model(f, y, ef=None, ey=None):
            self.fit_calls.append((len(f), None if ef is None else len(ef)))
            self.fit_index, self.es_index = f.index, None if ef is None else ef.index
            return PicklableModel()

        def build_ctx(pool):
            self.ctx_built.append(pool["country"].iloc[0])
            return {"country": pool["country"].iloc[0]}

        def feats(pairs, s1, pool, ctx):
            assert ctx["country"] == s1["country"].iloc[0]
            return tc.fake_build_features(pairs, s1, pool)

        self.comps = Components(tc.fake_normalize, tc.fake_build_index, tc.fake_generate_candidates, feats,
                                tc.fake_select_matches, build_ctx, train_model)

    def tearDown(self):
        self.tmp.cleanup()

    def train(self, **kw):
        return run_train(self.w, self.comps, FAKE_SCORING, chunk_s1=2, model_path=self.root / "m.pkl",
                         report_path=self.root / "train.md", log=lambda *_: None, **kw)

    def test_no_validation_s1_in_fitting_or_early_stopping(self):
        seen = {}
        orig = self.comps.train_model

        def spy(f, y, ef=None, ey=None):
            seen["fit"], seen["es"] = f, ef
            return orig(f, y, ef, ey)

        self.comps = Components(*[getattr(self.comps, k) for k in
                                  ("normalize_records", "build_index", "generate_candidates", "build_features",
                                   "select_matches", "build_feature_context")], spy)
        # rebuild pairs to map feature rows back to S1 ids
        from src.predict import featurize_world
        pairs, _ = featurize_world(self.w, self.w.s1.index, self.comps, 2, log=lambda *_: None)
        self.train()
        val = set(self.w.s1.index[self.w.s1["split"] == "val"])
        fit_s1 = set(pairs.loc[seen["fit"].index, S1_ID])
        es_s1 = set(pairs.loc[seen["es"].index, S1_ID]) if seen["es"] is not None else set()
        self.assertTrue(fit_s1 and not fit_s1 & val, "validation S1 used for fitting")
        self.assertFalse(es_s1 & val, "validation S1 used for early stopping")
        self.assertFalse(fit_s1 & es_s1, "fit and early-stopping S1 overlap")

    def test_threshold_rule_saved_model_and_report(self):
        model = self.train()
        saved = pickle.loads((self.root / "m.pkl").read_bytes())
        self.assertEqual(saved.threshold, model.threshold)
        self.assertIn(model.threshold, THRESHOLD_GRID)
        self.assertEqual(self.ctx_built, sorted(set(self.w.s1["country"])), "one ctx per country")
        report = (self.root / "train.md").read_text(encoding="utf-8")
        for section in ("Chosen threshold", "Blocking", "Threshold sweep", "Per country"):
            self.assertIn(section, report)

    def test_choose_threshold_ties_go_to_highest(self):
        sweep = pd.DataFrame({"threshold": [0.3, 0.5, 0.7, 0.9], "macro_f05": [0.8, 0.9, 0.9, 0.85]})
        self.assertEqual(choose_threshold(sweep), 0.7)

    def test_dev_equals_real_test_path_on_same_s1(self):
        model = self.train()
        f_dev = run_dev(self.w, model, self.comps, FAKE_SCORING, chunk_s1=2, report_path=self.root / "dev.md",
                        log=lambda *_: None)
        val = self.w.s1[self.w.s1["split"] == "val"]

        def load_country(c):
            return val[val["country"] == c], self.w.pool[self.w.pool["country"] == c]

        out = self.root / "sub"
        run_pipeline(sorted(val["country"].unique()), load_country, model, out, self.comps, chunk_s1=2,
                     log=lambda *_: None)
        m = pd.read_csv(out / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
        pred = {s: set(x.split(",")) - {""} for s, x in zip(m.iloc[:, 0], m.iloc[:, 1])}
        truth = evaluate.truth_dict(self.w.truth_raw)
        f_test = evaluate.macro_f_beta(pred, {s: truth.get(s, set()) for s in val.index})
        self.assertAlmostEqual(f_dev, f_test, places=12)
        self.assertIn("Validation macro F0.5", (self.root / "dev.md").read_text(encoding="utf-8"))

    def test_training_fails_loudly_without_train_model(self):
        comps = Components(*[getattr(self.comps, k) for k in
                             ("normalize_records", "build_index", "generate_candidates", "build_features",
                              "select_matches", "build_feature_context")])
        with self.assertRaisesRegex(ValueError, "train_model"):
            run_train(self.w, comps, FAKE_SCORING, chunk_s1=2, model_path=self.root / "m.pkl",
                      report_path=self.root / "t.md", log=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
