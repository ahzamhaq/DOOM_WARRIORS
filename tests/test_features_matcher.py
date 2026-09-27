"""Unit tests for features.py and matcher.py (Person 3). Run with the contract test:

    python -m unittest discover -s tests -t . -v
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BLK_COLUMNS, CAND_ID, PROBA, S1_ID
from src.features import FEATURE_NAMES, build_context, build_features
from src.matcher import Model, select_matches, train


def records(rows):
    df = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])
    return df.set_index("entity_id", drop=False)


S1 = records([
    ("S1-1", "Ram Marketing Pvt Ltd", "12 MG Rd, Delhi", "India"),
    ("S1-2", "Acme Plumbing LLC", "5 Oak St, Austin, TX", "US"),
    ("S1-3", "Zephyr Labs Inc", "", "US"),
])
POOL = records([
    ("S2-1", "राम मार्केटिंग प्राइवेट लिमिटेड", "12 MG ROAD, DELHI", "India"),
    ("S2-2", "ACME PLUMBING L.L.C.", "5 OAK STREET AUSTIN TX", "US"),
    ("S3-1", "Totally Different Bakery", "99 Pine Ave, Boston, MA", "US"),
    ("S3-2", "zephyrlabs.com", "", "US"),
])
PAIRS = pd.DataFrame({S1_ID: ["S1-1", "S1-2", "S1-2", "S1-3"], CAND_ID: ["S2-1", "S2-2", "S3-1", "S3-2"],
                      "blk_name_score": [0.4, 0.9, 0.1, 0.5], "blk_addr_score": [0.8, 0.7, np.nan, np.nan],
                      "blk_name_rank": [0, 0, 5, 1], "blk_addr_rank": [0, 1, -1, -1]},
                     index=[10, 20, 30, 40])  # non-default index on purpose


class FeatureTests(unittest.TestCase):
    def test_shape_index_dtype(self):
        f = build_features(PAIRS, S1, POOL)
        self.assertEqual(list(f.columns), FEATURE_NAMES)
        self.assertTrue(f.index.equals(PAIRS.index))
        self.assertTrue((f.dtypes == np.float32).all())
        self.assertFalse(np.isinf(f.to_numpy()).any())

    def test_blk_passed_through(self):
        f = build_features(PAIRS, S1, POOL)
        np.testing.assert_allclose(f[BLK_COLUMNS].to_numpy(), PAIRS[BLK_COLUMNS].to_numpy(dtype=np.float32))
        f2 = build_features(PAIRS[[S1_ID, CAND_ID]], S1, POOL)
        self.assertTrue(f2[BLK_COLUMNS].isna().all().all())

    def test_indic_name_is_transliterated(self):
        f = build_features(PAIRS, S1, POOL)
        self.assertGreater(f.loc[10, "name_token_set"], 0.5)  # Devanagari vs English spelling
        self.assertEqual(f.loc[10, "house_num"], 1.0)
        self.assertEqual(f.loc[20, "name_token_set"], 1.0)    # Pvt/LLC abbreviations normalised
        self.assertLess(f.loc[30, "name_token_set"], 0.5)
        self.assertEqual(f.loc[40, "domain_name"], 1.0)
        self.assertEqual(f.loc[40, "addr_missing"], 2.0)

    def test_chunking_does_not_change_features(self):
        full = build_features(PAIRS, S1, POOL)
        part = build_features(PAIRS.iloc[1:3], S1, POOL)
        pd.testing.assert_frame_equal(full.loc[part.index], part)

    def test_country_context_gives_identical_features(self):
        ctx = build_context(POOL)
        pd.testing.assert_frame_equal(build_features(PAIRS, S1, POOL, ctx=ctx), build_features(PAIRS, S1, POOL))
        part = PAIRS.iloc[1:3]  # a chunk reusing the country's ctx
        pd.testing.assert_frame_equal(build_features(part, S1, POOL, ctx=ctx), build_features(part, S1, POOL))

    def test_missing_id_raises(self):
        bad = PAIRS.copy()
        bad.loc[40, CAND_ID] = "S3-999"
        with self.assertRaises(KeyError):
            build_features(bad, S1, POOL)
        bad = PAIRS.copy()
        bad.loc[10, S1_ID] = "S1-999"
        with self.assertRaises(KeyError):
            build_features(bad, S1, POOL)

    def test_empty_pairs(self):
        f = build_features(PAIRS.iloc[:0], S1, POOL)
        self.assertEqual(list(f.columns), FEATURE_NAMES)
        self.assertEqual(len(f), 0)


class MatcherTests(unittest.TestCase):
    def test_select_matches_threshold_and_one_to_one(self):
        scored = pd.DataFrame({S1_ID: ["S1-2", "S1-1", "S1-3", "S1-1"],
                               CAND_ID: ["S2-9", "S2-9", "S2-9", "S3-1"],
                               PROBA: np.array([0.8, 0.8, 0.95, 0.4], dtype=np.float32)})
        off = select_matches(scored, 0.5, one_to_one=False)
        self.assertEqual(list(off.columns), [S1_ID, CAND_ID])
        self.assertEqual(len(off), 3)
        on = select_matches(scored, 0.5, one_to_one=True)
        self.assertEqual(on.values.tolist(), [["S1-3", "S2-9"]])
        tie = select_matches(scored[scored[S1_ID] != "S1-3"], 0.5, one_to_one=True)
        self.assertEqual(tie.values.tolist(), [["S1-1", "S2-9"]])  # tie -> smallest s1_id
        self.assertEqual(len(select_matches(scored.iloc[:0], 0.5)), 0)

    def test_model_roundtrip_and_column_order(self):
        rng = np.random.default_rng(0)
        feats = pd.DataFrame(rng.random((400, len(FEATURE_NAMES))).astype(np.float32), columns=FEATURE_NAMES)
        labels = pd.Series((feats["name_token_set"] > 0.5).astype(np.int8), index=feats.index)
        model = train(feats, labels, params={"n_estimators": 20, "min_child_samples": 5})
        model.threshold = 0.6
        p = model.predict_proba(feats)
        self.assertEqual(p.shape, (400,))
        self.assertEqual(p.dtype, np.float32)
        np.testing.assert_allclose(model.predict_proba(feats[FEATURE_NAMES[::-1]]), p)
        with tempfile.TemporaryDirectory() as d:
            model.save(Path(d) / "m.joblib")
            m2 = Model.load(Path(d) / "m.joblib")
        self.assertEqual(m2.threshold, 0.6)
        np.testing.assert_allclose(m2.predict_proba(feats), p)


if __name__ == "__main__":
    unittest.main()
