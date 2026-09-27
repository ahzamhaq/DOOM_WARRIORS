import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

import src.blocking as B
from src.blocking import build_index, generate_candidates
from src.config import BLK_COLUMNS, CAND_ID, S1_ID
from src.normalize import normalize_records


def recs(rows):
    df = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])
    return normalize_records(df.set_index("entity_id", drop=False))


POOL = recs([
    ("S2-1", "Goins & Greene Inc", "105 Mountain Lane, Cottontown, TN", "US"),
    ("S3-1", "Goins and Greene LLC", "105 Mountain Ln, Cottontown, Tennessee", "US"),
    ("S2-2", "Atlantic League LLC", "11513 Fallsburg Road, OH", "US"),
    ("S3-2", "श्री स्मार्ट प्रोजेक्ट्स प्राइवेट लिमिटेड", "11513 Fallsburg Road, OH", "US"),
    ("S2-3", "Novon Ship", "", "US"),
    ("S3-3", "", "", "US"),
    ("S2-4", "Goins Greene", "1 Rue X, Paris", "France"),
] + [(f"S2-{100 + i}", f"Greene Clone {i}", f"{i} Mountain Lane, TN", "US") for i in range(30)])

S1 = recs([
    ("S1-1", "Goins & Greene", "105 Mountain Lane, Cottontown, TN", "US"),
    ("S1-2", "Atlantic League", "11513 Fallsburg Rd, OH", "US"),
    ("S1-3", "", "", "US"),
    ("S1-4", "Novon Ship Inc", "", "US"),
])


def cands(pairs):
    return {s: set(g) for s, g in pairs.groupby(S1_ID)[CAND_ID]}


class TestBlocking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = build_index(POOL)
        cls.pairs = generate_candidates(S1, cls.index, k=5, max_cands=8)

    def test_shape_and_types(self):
        p = self.pairs
        self.assertEqual(list(p.columns), [S1_ID, CAND_ID, *BLK_COLUMNS])
        self.assertTrue(p[CAND_ID].str.match(r"S[23]-").all())
        for c in BLK_COLUMNS:
            self.assertTrue(np.issubdtype(p[c].dtype, np.number))

    def test_finds_true_matches_by_name_and_address(self):
        c = cands(self.pairs)
        self.assertTrue({"S2-1", "S3-1"} <= c["S1-1"])
        self.assertTrue({"S2-2", "S3-2"} <= c["S1-2"])  # S3-2 only reachable via address
        self.assertIn("S2-3", c["S1-4"])  # empty address: name retriever still works

    def test_missing_retriever_convention(self):
        p = self.pairs.set_index([S1_ID, CAND_ID])
        row = p.loc[("S1-2", "S3-2")]
        self.assertTrue(np.isnan(row["blk_name_score"]) or row["blk_name_score"] < 0.1)
        self.assertGreaterEqual(row["blk_addr_rank"], 0)
        miss_n = p["blk_name_rank"] == -1
        self.assertTrue(p.loc[miss_n, "blk_name_score"].isna().all())
        miss_a = p["blk_addr_rank"] == -1
        self.assertTrue(p.loc[miss_a, "blk_addr_score"].isna().all())
        self.assertTrue(p.loc[~miss_n, "blk_name_score"].gt(0).all())
        self.assertTrue(p.loc[~miss_a, "blk_addr_score"].gt(0).all())

    def test_empty_s1_row_has_no_candidates(self):
        self.assertNotIn("S1-3", set(self.pairs[S1_ID]))

    def test_same_country_only_and_no_duplicates(self):
        self.assertNotIn("S2-4", set(self.pairs[CAND_ID]))
        self.assertFalse(self.pairs.duplicated([S1_ID, CAND_ID]).any())

    def test_cap_respected(self):
        self.assertLessEqual(self.pairs.groupby(S1_ID).size().max(), 8)
        tight = generate_candidates(S1, self.index, k=5, max_cands=3)
        self.assertLessEqual(tight.groupby(S1_ID).size().max(), 3)
        self.assertLessEqual(generate_candidates(S1, self.index, k=2, max_cands=40).groupby(S1_ID).size().max(), 4)

    def test_chunking_and_repeat_do_not_change_output(self):
        chunked = pd.concat([generate_candidates(S1.iloc[i:i + 1], self.index, k=5, max_cands=8)
                             for i in range(len(S1))], ignore_index=True)
        pd.testing.assert_frame_equal(chunked, self.pairs)
        pd.testing.assert_frame_equal(generate_candidates(S1, build_index(POOL), k=5, max_cands=8), self.pairs)

    def test_pool_order_does_not_matter(self):
        shuffled = build_index(POOL.sample(frac=1, random_state=3))
        pd.testing.assert_frame_equal(generate_candidates(S1, shuffled, k=5, max_cands=8), self.pairs)

    def test_degenerate_inputs(self):
        empty_pool = build_index(POOL[POOL["entity_id"] == "S3-3"])  # only an all-empty record
        self.assertEqual(len(generate_candidates(S1, empty_pool)), 0)
        out = generate_candidates(S1.iloc[:0], self.index)
        self.assertEqual(list(out.columns), [S1_ID, CAND_ID, *BLK_COLUMNS])
        self.assertEqual(len(out), 0)


class TestStage1(unittest.TestCase):
    def test_top_per_row_matches_brute_force_with_ties(self):
        rng = np.random.default_rng(0)
        dense = rng.integers(0, 4, size=(30, 50)).astype(np.float32)  # many ties, many zeros
        rows, cols, scores = B._top_per_row(sp.csr_matrix(dense), 7)
        for i in range(dense.shape[0]):
            nz = [(-dense[i, j], j) for j in range(dense.shape[1]) if dense[i, j] > 0]
            expect = [j for _, j in sorted(nz)[:7]]
            self.assertEqual(sorted(cols[rows == i].tolist()), sorted(expect))
            self.assertTrue(np.all(scores[rows == i] > 0))

    def test_frequent_tokens_not_indexed(self):
        pool = recs([(f"S2-{i:03d}", f"Common Traders {i}", "Main Bazar, Pune", "India") for i in range(20)]
                    + [("S3-999", "Zephyr Exports", "Main Bazar, Pune", "India")])
        s1 = recs([("S1-1", "Zephyr Traders", "Main Bazar, Pune", "India")])
        old = B._TOKEN_MAX_DF
        B._TOKEN_MAX_DF = 5
        try:
            index = build_index(pool)
            score = sum(B._TOKENS.transform(s1[c]) @ index.fields[f].tokens_t for f, c in B._FIELDS)
        finally:
            B._TOKEN_MAX_DF = old
        hits = set(index.ids.take(score.tocsr().indices).to_pylist())
        self.assertEqual(hits, {"S3-999"})  # "traders"/"main bazar"/"pune" are in >5 records, "zephyr" is not

    def test_shortlist_bounds_candidates(self):
        old = B._SHORTLIST
        B._SHORTLIST = 3
        try:
            pairs = generate_candidates(S1, build_index(POOL), k=20, max_cands=40)
        finally:
            B._SHORTLIST = old
        self.assertLessEqual(pairs.groupby(S1_ID).size().max(), 3)

    def test_reordered_address_and_indic_name_found(self):
        pool = recs([("S2-1", "राम फाउंडेशन लिमिटेड", "PATNA, बिहार, C/O S.K.TRADING CO. BUDH MARG", "India")]
                    + [(f"S3-{i}", f"Ram Foundation Branch {i}", f"{i} Other Road, Patna", "India") for i in range(40)])
        s1 = recs([("S1-1", "Ram Foundation Limited", "C/O S.K. Trading Co. Budh Marg, Patna, Bihar", "India")])
        self.assertIn("S2-1", set(generate_candidates(s1, build_index(pool), k=5, max_cands=10)[CAND_ID]))

    def test_stage1_list_is_third_retriever(self):
        pool = recs([
            ("S2-1", "Zephyr Exports Kiln", "Other Road, Delhi", "India"),        # best name
            ("S2-2", "Unrelated Foo", "Kiln Street, Pune, Maharashtra", "India"),  # best address
            ("S2-3", "Zephyr Exports", "Kiln Street, Pune", "India"),             # best combined word score
        ])
        s1 = recs([("S1-1", "Zephyr Exports Kiln", "Kiln Street, Pune, Maharashtra", "India")])
        p = generate_candidates(s1, build_index(pool), k=1, max_cands=40).set_index(CAND_ID)
        self.assertEqual(set(p.index), {"S2-1", "S2-2", "S2-3"})
        self.assertEqual(p.loc["S2-1", "blk_name_rank"], 0)
        self.assertEqual(p.loc["S2-2", "blk_addr_rank"], 0)
        self.assertEqual(p.loc[["S2-3"], ["blk_name_rank", "blk_addr_rank"]].values.tolist(), [[-1, -1]])
        self.assertTrue(p.loc[["S2-3"], ["blk_name_score", "blk_addr_score"]].isna().all().all())


TEXTS = pd.Series(["goins and greene", "", "श्री स्मार्ट प्रोजेक्ट्स", "aaa aaa aaa", "105 mountain ln cottontown tn",
                   "goins greene", "x", "zephyr exports kiln", "105 mountain ln", "", "b3 626 v a lucknow"] * 3)


class TestSlicedBuildIsExact(unittest.TestCase):
    """The sliced (low-memory) builders must equal the whole-pool formulas bit for bit."""

    def setUp(self):
        self.old_slice = B._BUILD_SLICE
        B._BUILD_SLICE = 4  # many slice boundaries

    def tearDown(self):
        B._BUILD_SLICE = self.old_slice

    def assert_same(self, a, b):
        for attr in ("indptr", "indices", "data"):
            self.assertTrue(np.array_equal(getattr(a, attr), getattr(b, attr)), attr)
        self.assertEqual(a.shape, b.shape)

    def test_char_tfidf_equals_sklearn_fit_transform(self):
        ref = B.TfidfVectorizer(analyzer="char_wb", ngram_range=B._CHAR_NGRAM, lowercase=False, sublinear_tf=True,
                                dtype=np.float32)
        expect = ref.fit_transform(TEXTS).tocsr()
        vec, got = B._char_tfidf(TEXTS)
        self.assert_same(got, expect)
        self.assertEqual(vec.vocabulary_, ref.vocabulary_)
        self.assertTrue(np.array_equal(vec.idf_, ref.idf_))
        self.assert_same(vec.transform(TEXTS[:7]), ref.transform(TEXTS[:7]))

    def test_token_index_equals_whole_matrix_formula(self):
        x = B._TOKENS.transform(TEXTS)
        df = np.bincount(x.indices, minlength=x.shape[1])
        w = x @ sp.diags((np.log((x.shape[0] + 1) / (df + 1)) + 1).astype(np.float32))
        norm = np.sqrt(np.asarray(w.multiply(w).sum(1)).ravel())
        old = B._TOKEN_MAX_DF
        B._TOKEN_MAX_DF = 5  # prune some tokens
        try:
            w = sp.diags((1 / np.where(norm > 0, norm, 1)).astype(np.float32)) @ w @ sp.diags((df <= 5).astype(np.float32))
            w.eliminate_zeros()
            self.assert_same(B._token_index(TEXTS), w.T.tocsr())
        finally:
            B._TOKEN_MAX_DF = old


if __name__ == "__main__":
    unittest.main()
