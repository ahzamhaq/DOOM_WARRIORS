"""Build the final submission zip and refuse to build it if anything is wrong.  Owner: Person 4.

    python -m scripts.package_submission                      # outputs/*.tsv -> outputs/submission.zip
    python -m scripts.package_submission --light              # --check-ids run without -c (low RAM)
    python -m scripts.package_submission --allow-placeholders # dry run while the doc still has [[TBD ...]]

Zip layout (CLAUDE.md "Output requirements"; confirm against data/raw/README.md before the final upload):
    output/matching_results.tsv
    output/candidate_pairs.tsv
    code/business_entity_resolution/{src,scripts,notebooks}/*.py, README.md, requirements.txt
    Documentation_template.md

Checks, all before the zip is written (a failed run leaves no zip behind):
  1. both output files: exact headers, same S1 rows in the same order, no duplicate S1, S2-/S3- ids only,
     no duplicate id in a list, matches a subset of candidates, one row per test S1;
  2. requirements.txt fully pinned (==); no network access / URLs in the packaged code (external-data review);
  3. no `[[TBD` placeholder left in the documentation or the code README;
  4. the official validator (data/raw/utils/validate_submission.py), first plain and then with --check-ids,
     must exit 0 and print PASS.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from src.config import CANDIDATE_FILE, DATA_DIR, MATCHING_FILE, OUTPUT_DIR, ROOT

VALIDATOR = ROOT / "data" / "raw" / "utils" / "validate_submission.py"
CODE_ROOT = "code/business_entity_resolution"
CODE_DIRS = ("src", "scripts", "notebooks")
PLACEHOLDER = "[[TBD"
HEADERS = {MATCHING_FILE: "source1_entity_id\tmatched_entity_ids",
           CANDIDATE_FILE: "source1_entity_id\tcandidate_entity_ids"}
# External-data review: the code must not reach the network. Written so this file does not match itself.
NETWORK = re.compile(r"(?:ht" r"tps?://|^\s*(?:import|from)\s+(?:requests|urllib|httpx|aiohttp|socket|ftplib)\b)",
                     re.MULTILINE)
ZIP_TIME = (2026, 1, 1, 0, 0, 0)  # fixed timestamps: same inputs -> same zip bytes


class PackagingError(Exception):
    pass


def _ids(field: str) -> list[str]:
    return field.split(",") if field else []


def check_outputs(matches: Path, candidates: Path, test_s1: Path | None) -> int:
    """Stream both files in lockstep (never all in RAM). Returns the number of S1 rows."""
    problems: list[str] = []

    def bad(msg: str) -> None:
        if len(problems) < 20:
            problems.append(msg)

    seen: set[str] = set()
    n = 0
    with open(matches, encoding="utf-8", newline="") as fm, open(candidates, encoding="utf-8", newline="") as fc:
        for name, f in ((MATCHING_FILE, fm), (CANDIDATE_FILE, fc)):
            head = f.readline().rstrip("\r\n")
            if head != HEADERS[name]:
                raise PackagingError(f"{name}: header {head!r}, expected {HEADERS[name]!r}")
        for n, (lm, lc) in enumerate(zip(fm, fc, strict=False), start=1):
            rm, rc = lm.rstrip("\r\n").split("\t"), lc.rstrip("\r\n").split("\t")
            if len(rm) != 2 or len(rc) != 2:
                bad(f"row {n}: expected 2 tab-separated fields")
                continue
            if rm[0] != rc[0]:
                raise PackagingError(f"row {n}: S1 order differs ({rm[0]} vs {rc[0]}); both files must list the "
                                     "same S1 ids in the same order")
            s1 = rm[0]
            if not s1.startswith("S1-"):
                bad(f"row {n}: {s1!r} is not an S1 id")
            if s1 in seen:
                bad(f"{s1}: duplicate row")
            seen.add(s1)
            m, c = _ids(rm[1]), _ids(rc[1])
            for name, lst in (("matched", m), ("candidate", c)):
                if len(set(lst)) != len(lst):
                    bad(f"{s1}: duplicate id in {name} list")
                if any(not x.startswith(("S2-", "S3-")) for x in lst):
                    bad(f"{s1}: {name} list has a non S2-/S3- id")
            if not set(m) <= set(c):
                bad(f"{s1}: matches not a subset of candidates ({sorted(set(m) - set(c))[:3]})")
        if fm.readline() or fc.readline():
            raise PackagingError("the two output files have a different number of rows")
    if test_s1 is not None:
        with open(test_s1, encoding="utf-8") as f:
            n_test = sum(1 for _ in f) - 1
        if n_test != n:
            bad(f"{n:,} S1 rows but {n_test:,} S1 entities in {test_s1.name}")
    if problems:
        raise PackagingError("output files:\n  - " + "\n  - ".join(problems))
    return n


def code_files(repo: Path) -> list[Path]:
    return sorted(p for d in CODE_DIRS for p in (repo / d).rglob("*.py") if "__pycache__" not in p.parts)


def check_code(repo: Path, requirements: Path) -> None:
    problems = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line and "==" not in line:
            problems.append(f"requirements.txt: {line!r} is not pinned (use ==)")
    for p in code_files(repo):
        for m in NETWORK.finditer(p.read_text(encoding="utf-8")):
            problems.append(f"{p.relative_to(repo)}: network access / URL {m.group(0).strip()!r}")
    if problems:
        raise PackagingError("code review:\n  - " + "\n  - ".join(problems))


def check_docs(docs: list[Path], allow_placeholders: bool) -> None:
    left = [f"{d.name}:{i}: {line.strip()[:90]}" for d in docs
            for i, line in enumerate(d.read_text(encoding="utf-8").splitlines(), start=1) if PLACEHOLDER in line]
    if left and not allow_placeholders:
        raise PackagingError(f"{len(left)} unfilled {PLACEHOLDER} placeholder(s):\n  - " + "\n  - ".join(left[:20]))


def run_validator(validator: Path, matches: Path, candidates: Path, test_dir: Path, light: bool,
                  log=print) -> None:
    runs = [["-m", str(matches), "-c", str(candidates), "-t", str(test_dir)],
            ["-m", str(matches), *([] if light else ["-c", str(candidates)]), "-t", str(test_dir), "--check-ids"]]
    for args in runs:
        log(f"validator: {' '.join(a if a.startswith('-') else Path(a).name for a in args)}")
        r = subprocess.run([sys.executable, str(validator), *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        out = (r.stdout + r.stderr).strip()
        log("  " + out.replace("\n", "\n  "))
        if r.returncode != 0 or "PASS" not in r.stdout:
            raise PackagingError(f"official validator failed (exit {r.returncode}):\n{out}")


def build_zip(out: Path, entries: dict[str, Path]) -> None:
    tmp = out.with_suffix(".zip.partial")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for arc, src in sorted(entries.items()):
                info = zipfile.ZipInfo(arc, date_time=ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                with open(src, "rb") as f, z.open(info, "w", force_zip64=True) as dst:
                    while block := f.read(1 << 20):
                        dst.write(block)
        with zipfile.ZipFile(tmp) as z:  # re-read: CRC of every member + exact member list
            badfile = z.testzip()
            if badfile or sorted(z.namelist()) != sorted(entries):
                raise PackagingError(f"zip verification failed ({badfile or 'member list differs'})")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)


def package(matches: Path, candidates: Path, doc: Path, readme: Path, out: Path, test_dir: Path,
            validator: Path = VALIDATOR, repo: Path = ROOT, light: bool = False, allow_placeholders: bool = False,
            log=print) -> dict[str, Path]:
    """Run every check, then write the zip. Raises PackagingError (and writes nothing) on any problem."""
    if out.exists():  # never leave an older zip around that could be uploaded by mistake after a failed run
        out.unlink()
        log(f"removed previous {out.name}")
    requirements = repo / "requirements.txt"
    need = {"matching results": matches, "candidate pairs": candidates, "documentation": doc,
            "code README": readme, "requirements.txt": requirements, "official validator": validator,
            "test data dir": test_dir, "src/": repo / "src"}
    missing = [f"{k}: {v}" for k, v in need.items() if not v.exists()]
    if missing:
        raise PackagingError("missing inputs:\n  - " + "\n  - ".join(missing))
    test_s1 = test_dir / "test_source1.tsv"

    n = check_outputs(matches, candidates, test_s1 if test_s1.exists() else None)
    log(f"outputs: {n:,} S1 rows, headers / ids / matches-subset-of-candidates OK")
    check_code(repo, requirements)
    check_docs([doc, readme], allow_placeholders)
    log("code + docs: requirements pinned, no network access, placeholders "
        + ("allowed (dry run)" if allow_placeholders else "none"))
    run_validator(validator, matches, candidates, test_dir, light, log)

    entries = {f"output/{MATCHING_FILE}": matches, f"output/{CANDIDATE_FILE}": candidates,
               f"{CODE_ROOT}/README.md": readme, f"{CODE_ROOT}/requirements.txt": requirements,
               "Documentation_template.md": doc}
    entries.update({f"{CODE_ROOT}/{p.relative_to(repo).as_posix()}": p for p in code_files(repo)})
    build_zip(out, entries)
    log(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MB, {len(entries)} files)")
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", type=Path, default=OUTPUT_DIR / MATCHING_FILE)
    ap.add_argument("--candidates", type=Path, default=OUTPUT_DIR / CANDIDATE_FILE)
    ap.add_argument("--doc", type=Path, default=ROOT / "docs" / "Documentation_template.md")
    ap.add_argument("--readme", type=Path, default=ROOT / "docs" / "submission_README.md")
    ap.add_argument("--test-dir", type=Path, default=DATA_DIR / "test")
    ap.add_argument("--validator", type=Path, default=VALIDATOR)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "submission.zip")
    ap.add_argument("--light", action="store_true", help="--check-ids run without -c (validator RAM)")
    ap.add_argument("--allow-placeholders", action="store_true", help="dry run: accept [[TBD ...]] in the docs")
    a = ap.parse_args()
    try:
        package(a.matches, a.candidates, a.doc, a.readme, a.out, a.test_dir, a.validator, ROOT, a.light,
                a.allow_placeholders)
    except PackagingError as e:
        print(f"\nPACKAGING FAILED, no zip written:\n{e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
