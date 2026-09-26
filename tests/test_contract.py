"""Toy end-to-end CONTRACT test (interfaces and data flow only; says nothing about ML quality).

    python -m unittest discover -s tests -t . -v

Fake normalize / blocking / features / model / select_matches are injected into the SAME `run_pipeline`
that produces the real submission, then the official validator is run on the toy output. A real module
that keeps to docs/CONTRACT.md can replace its fake here without touching anything else.

Toy world: 4 countries (one, "Atlantis", is unseen: nothing may hard-code the country list), 6 S1 each:
  0 singleton | 1 & 2 share ONE S2 record (truly ga's, but it also looks like gb's) | 3 one S2 match
  4 S2+S3 match | 5 singleton.   The shared record is what the one-to-one option is judged on.
"""
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BLK_COLUMNS, CAND_ID, NORM_COLUMNS, ONE_TO_ONE, PROBA, ROOT, S1_ID
from src.evaluate import macro_f_beta
from src.predict import Components, run_pipeline

VALIDATOR = ROOT / "data" / "raw" / "utils" / "validate_submission.py"
COUNTRIES = ["US", "India", "France", "Atlantis"]
COLS = ["entity_id", "business_name", "business_address", "country"]
STOP = {"traders", "inc", "ltd", "holdings"}


# ------------------------------------------------------------------ toy data
def build_toy():
    s1, s2, s3, truth = [], [], [], {}
    counters = {"S2": 0, "S3": 0}
    g = 0

    def rec(rows, src, name, addr, c):
        counters[src] += 1
        rid = f"{src}-{counters[src]:03d}"
        rows.append((rid, name, addr, c))
        return rid

    for c in COUNTRIES:
        ga = None
        for i in range(6):
            g += 1
            sid, tag = f"S1-{g:03d}", f"{g:03d}"
            s1.append((sid, f"Biz{tag} Traders Inc", f"{tag} Main Street", c))
            truth[sid] = set()
            if i == 1:  # ga: its only match is the shared record (created when we reach i == 2)
                ga = (sid, tag)
            if i == 2:
                shared = rec(s2, "S2", f"BIZ{ga[1]} BIZ{tag} LTD", f"{ga[1]} MAIN STREET", c)
                truth[ga[0]].add(shared)
            if i in (2, 3, 4):
                truth[sid].add(rec(s2, "S2", f"BIZ{tag} TRADERS LTD", f"{tag} MAIN STREET", c))
            if i in (2, 4):
                truth[sid].add(rec(s3, "S3", f"Biz{tag} Trdrs", "", c))  # empty address on purpose
        for n in range(3):
            rec(s2, "S2", f"Zzz{c}{n} Holdings", "1 Nowhere Rd", c)
        for n in range(2):
            rec(s3, "S3", f"Yyy{c}{n} Holdings", "", c)
    return [pd.DataFrame(r, columns=COLS) for r in (s1, s2, s3)], truth


# ------------------------------------------------------------------ fake components
def _tokens(s):
    return set(re.findall(r"[a-z0-9]+", s.lower())) - STOP


def fake_normalize(df):
    out = df.copy()
    out["name_norm"] = df["business_name"].str.lower()
    out["addr_norm"] = df["business_address"].str.lower()
    out["name_is_domain"] = False
    out["name_script"] = "latin"
    out["addr_empty"] = df["business_address"] == ""
    out["house_no"] = ""
    return out


def fake_build_index(pool):
    index = {}
    for rid, name in zip(pool["entity_id"], pool["name_norm"]):
        for t in _tokens(name):
            index.setdefault(t, []).append(rid)
    return index


def fake_generate_candidates(s1, index, k=20, max_cands=40):
    rows = []
    for sid, name in zip(s1["entity_id"], s1["name_norm"]):
        toks = _tokens(name)
        score = {}
        for t in toks:
            for cid in index.get(t, ()):
                score[cid] = score.get(cid, 0.0) + 1 / len(toks)
        for rank, (cid, sc) in enumerate(sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:max_cands]):
            rows.append((sid, cid, sc, np.nan, rank, -1))
    return pd.DataFrame(rows, columns=[S1_ID, CAND_ID, *BLK_COLUMNS])


def fake_build_features(pairs, s1, pool):
    a, b = s1.loc[pairs[S1_ID]], pool.loc[pairs[CAND_ID]]  # relies on the "indexed by entity_id" contract
    jac = [len(_tokens(x) & _tokens(y)) / len(_tokens(x) | _tokens(y)) for x, y in zip(a["name_norm"], b["name_norm"])]
    same_addr = (a["addr_norm"].to_numpy() == b["addr_norm"].to_numpy()) & (a["addr_norm"].to_numpy() != "")
    feats = pd.DataFrame({"name_jaccard": jac, "addr_equal": same_addr, **{c: pairs[c] for c in BLK_COLUMNS}},
                         index=pairs.index)
    return feats.astype("float32")


class FakeModel:
    threshold = 0.35

    def predict_proba(self, feats):
        return (0.8 * feats["name_jaccard"] + 0.2 * feats["addr_equal"]).to_numpy(np.float32)


def fake_select_matches(scored, threshold, one_to_one=False):
    m = scored[scored[PROBA] >= threshold]
    if one_to_one:
        m = m.sort_values([PROBA, S1_ID], ascending=[False, True]).drop_duplicates(CAND_ID)
    return m[[S1_ID, CAND_ID]]


FAKES = Components(fake_normalize, fake_build_index, fake_generate_candidates, fake_build_features,
                   fake_select_matches)


# ------------------------------------------------------------------ tests
def read_lists(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df, {r[0]: set(r[1].split(",")) if r[1] else set() for r in df.itertuples(index=False)}


class ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        (cls.s1, cls.s2, cls.s3), cls.truth = build_toy()
        cls.test_dir = cls.root / "test"
        cls.test_dir.mkdir()
        for n, df in (("1", cls.s1), ("2", cls.s2), ("3", cls.s3)):
            df.to_csv(cls.test_dir / f"test_source{n}.tsv", sep="\t", index=False, lineterminator="\n")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def load_country(self, c):
        idx = lambda d: d[d["country"] == c].set_index("entity_id", drop=False)  # noqa: E731
        return idx(self.s1), pd.concat([idx(self.s2), idx(self.s3)])

    def run_toy(self, name, one_to_one, chunk_s1):
        out = self.root / name
        run_pipeline(COUNTRIES, self.load_country, FakeModel(), out, FAKES, chunk_s1, one_to_one, log=lambda *_: None)
        return out

    def validate(self, out):
        if not VALIDATOR.exists():
            self.skipTest(f"official validator not found at {VALIDATOR} (unzip student_resource into data/raw/)")
        r = subprocess.run(
            [sys.executable, str(VALIDATOR), "-m", str(out / "matching_results.tsv"),
             "-c", str(out / "candidate_pairs.tsv"), "-t", str(self.test_dir), "--check-ids"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("PASS", r.stdout)

    # --- output files
    def check_files(self, out):
        for f, col in (("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")):
            df, _ = read_lists(out / f)
            self.assertEqual(list(df.columns), ["source1_entity_id", col])
            self.assertEqual(len(df), len(self.s1), "exactly one row per S1 entity")
            self.assertEqual(set(df["source1_entity_id"]), set(self.s1["entity_id"]))
        _, matches = read_lists(out / "matching_results.tsv")
        _, cands = read_lists(out / "candidate_pairs.tsv")
        for s, m in matches.items():
            self.assertTrue(m <= cands[s], f"{s}: matches must be a subset of candidates")
            self.assertTrue(all(x.startswith(("S2-", "S3-")) for x in m))
        return matches, cands

    def test_flow_with_one_to_one_off(self):
        out = self.run_toy("off", one_to_one=False, chunk_s1=4)  # 6 S1 per country -> 2 chunks each
        matches, cands = self.check_files(out)
        self.assertEqual(sum(1 for m in matches.values() if not m), 8, "2 singletons x 4 countries, empty rows kept")
        singletons = [s for s, t in self.truth.items() if not t]
        self.assertTrue(all(cands[s] == set() for s in singletons), "no-candidate S1 still gets an empty row")
        shared = [c for c in set.union(*matches.values()) if sum(c in m for m in matches.values()) > 1]
        self.assertEqual(len(shared), 4, "flag off: shared record is matched to two S1 in each country")
        self.assertLess(macro_f_beta(matches, self.truth), 1.0)
        atl = self.s1[self.s1["country"] == "Atlantis"]["entity_id"]
        self.assertTrue(any(matches[s] for s in atl), "unseen country must be processed, not dropped")
        self.validate(out)

    def test_flow_with_one_to_one_on(self):
        out = self.run_toy("on", one_to_one=True, chunk_s1=4)
        matches, _ = self.check_files(out)
        all_ids = [c for m in matches.values() for c in m]
        self.assertEqual(len(all_ids), len(set(all_ids)), "one-to-one: each S2/S3 id used at most once")
        self.assertAlmostEqual(macro_f_beta(matches, self.truth), 1.0)
        self.validate(out)

    def test_chunking_does_not_change_output(self):
        a, b = self.run_toy("c2", False, 2), self.run_toy("c100", False, 100)
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            self.assertEqual((a / f).read_bytes(), (b / f).read_bytes())

    def test_module_output_shapes(self):
        s1, pool = (fake_normalize(d) for d in self.load_country("France"))
        self.assertTrue(set(NORM_COLUMNS) <= set(s1.columns))
        pairs = fake_generate_candidates(s1, fake_build_index(pool))
        self.assertEqual(list(pairs.columns), [S1_ID, CAND_ID, *BLK_COLUMNS])
        self.assertFalse(pairs.duplicated([S1_ID, CAND_ID]).any())
        self.assertTrue(pairs[CAND_ID].str.startswith(("S2-", "S3-")).all())
        self.assertTrue((s1.loc[pairs[S1_ID], "country"].to_numpy() == pool.loc[pairs[CAND_ID], "country"].to_numpy()).all())
        feats = fake_build_features(pairs, s1, pool)
        self.assertTrue(feats.index.equals(pairs.index))
        self.assertTrue(all(np.issubdtype(t, np.floating) for t in feats.dtypes))
        self.assertTrue(np.isfinite(feats.fillna(0).to_numpy()).all())
        self.assertEqual(FakeModel().predict_proba(feats).shape, (len(pairs),))

    def test_one_to_one_is_off_by_default(self):
        self.assertFalse(ONE_TO_ONE, "experimental: only turn on after validation shows it helps")


if __name__ == "__main__":
    unittest.main()
