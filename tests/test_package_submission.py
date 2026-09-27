"""scripts/package_submission.py on toy outputs (layout, refusals, self-check)."""
import tempfile
import unittest
import zipfile
from pathlib import Path

import tests.test_contract as tc
from scripts import package_submission as ps

EXPECTED_TOP = {"output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
                f"{ps.PKG}/README.md", f"{ps.PKG}/requirements.txt"}


@unittest.skipUnless(ps.VALIDATOR.exists(), "official validator not unzipped into data/raw/utils/")
class PackageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (s1, s2, s3), self.truth = tc.build_toy()
        self.test_dir = root / "test"; self.test_dir.mkdir()
        for n, d in (("1", s1), ("2", s2), ("3", s3)):
            d.to_csv(self.test_dir / f"test_source{n}.tsv", sep="\t", index=False, lineterminator="\n")
        self.s1_ids = list(s1["entity_id"])
        self.out = root / "outputs"; self.out.mkdir()
        self.doc = root / "doc.md"
        self.doc.write_text("# Methodology\nTeam: DOOM_WARRIORS\nF0.5: 0.94\n", encoding="utf-8")
        self.zip = root / "sub.zip"

    def tearDown(self):
        self.tmp.cleanup()

    def write_outputs(self, with_matches=True, drop_last=False):
        ids = self.s1_ids[:-1] if drop_last else self.s1_ids
        rows = [(s, ",".join(sorted(self.truth[s])) if with_matches else "") for s in ids]
        for name, col in (("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")):
            with open(self.out / name, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"source1_entity_id\t{col}\n" + "".join(f"{s}\t{m}\n" for s, m in rows))

    def run_pkg(self, **kw):
        return ps.package(out_dir=self.out, test_dir=self.test_dir, doc=kw.pop("doc", self.doc), zip_path=self.zip,
                          include_model=False, log=lambda *_: None, **kw)

    def test_valid_package_layout_and_contents(self):
        self.write_outputs()
        self.run_pkg()
        with zipfile.ZipFile(self.zip) as z:
            names = set(z.namelist())
            self.assertTrue(EXPECTED_TOP <= names, EXPECTED_TOP - names)
            src = {n for n in names if n.startswith(f"{ps.PKG}/src/")}
            for m in ps.MODULES:
                self.assertIn(f"{ps.PKG}/src/{m}.py", src)
            self.assertIn(f"{ps.PKG}/src/__init__.py", src)
            self.assertEqual(z.read("output/matching_results.tsv"), (self.out / "matching_results.tsv").read_bytes())
            self.assertEqual(z.read("Documentation_template.md"), self.doc.read_bytes())
            self.assertFalse([n for n in names if n.endswith(".tsv") and not n.startswith("output/")])

    def test_refuses_all_empty_matches_unless_allowed(self):
        self.write_outputs(with_matches=False)
        with self.assertRaisesRegex(ps.PackagingError, "empty"):
            self.run_pkg()
        self.run_pkg(allow_empty=True)

    def test_refuses_when_official_validator_fails(self):
        self.write_outputs(drop_last=True)  # one S1 row missing -> validator FAIL
        with self.assertRaisesRegex(ps.PackagingError, "validator FAILED"):
            self.run_pkg()
        self.assertFalse(self.zip.exists())

    def test_refuses_unfilled_or_missing_doc_unless_draft_allowed(self):
        self.write_outputs()
        self.doc.write_text("**Team Name:** [Your Team Name]\n", encoding="utf-8")
        with self.assertRaisesRegex(ps.PackagingError, "placeholders"):
            self.run_pkg()
        self.run_pkg(allow_draft_doc=True)
        with self.assertRaisesRegex(ps.PackagingError, "not found"):
            self.run_pkg(doc=self.doc.with_name("missing.md"))


if __name__ == "__main__":
    unittest.main()
