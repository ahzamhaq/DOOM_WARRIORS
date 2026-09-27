"""Build the final submission zip.  Owner: Lead.

    python -m scripts.package_submission                 # validate outputs, build + self-check the zip
    python -m scripts.package_submission --check-ids     # also check every ID exists (uses a few GB RAM)

Layout (data/raw/README.md, "Final Submission Package"):
    DOOM_WARRIORS_submission.zip
    ├── output/matching_results.tsv, candidate_pairs.tsv        <- outputs/
    ├── code/business_entity_resolution/
    │   ├── src/            <- src/*.py + scripts/build_dev_world.py (all source under src/)
    │   ├── README.md       <- docs/SUBMISSION_README.md
    │   ├── requirements.txt (must be fully pinned)
    │   └── outputs/model.joblib (if present: reproduce without retraining)
    └── Documentation_template.md   <- docs/Documentation_template.md (the filled-in copy)

Refuses to build when: the official validator FAILs; every matched list is empty (a --dry-run output)
unless --allow-empty; the methodology doc is missing or still has template placeholders unless
--allow-draft-doc; requirements.txt has an unpinned line. After building, it unzips to a temp folder and
checks that the packaged code imports on its own and that nothing but the allowed files is inside.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from src.config import CANDIDATE_FILE, DATA_DIR, MATCHING_FILE, MODEL_PATH, OUTPUT_DIR, ROOT

TEAM = "DOOM_WARRIORS"
PKG = "code/business_entity_resolution"
VALIDATOR = ROOT / "data" / "raw" / "utils" / "validate_submission.py"
DOC = ROOT / "docs" / "Documentation_template.md"
README = ROOT / "docs" / "SUBMISSION_README.md"
EXTRA_SRC = [ROOT / "scripts" / "build_dev_world.py"]  # scripts the reproduction steps need
PLACEHOLDERS = ["[Your Team Name]", "[List all team members]", "[Date]", "[your best validation score]"]
MODULES = ["config", "data_loader", "normalize", "blocking", "features", "matcher", "evaluate", "predict",
           "build_dev_world"]


class PackagingError(Exception):
    pass


def validate_outputs(out_dir: Path, test_dir: Path, check_ids: bool) -> str:
    cmd = [sys.executable, str(VALIDATOR), "-m", str(out_dir / MATCHING_FILE),
           "-c", str(out_dir / CANDIDATE_FILE), "-t", str(test_dir)] + (["--check-ids"] if check_ids else [])
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # validator prints non-ASCII; Windows consoles default to cp1252
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if r.returncode != 0 or "PASS" not in r.stdout:
        raise PackagingError("official validator FAILED:\n" + r.stdout + r.stderr)
    return r.stdout


def count_nonempty(path: Path) -> tuple[int, int]:
    rows = nonempty = 0
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            rows += 1
            nonempty += bool(line.rstrip("\n").partition("\t")[2])
    return rows, nonempty


def check_requirements(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    loose = [ln for ln in lines if "==" not in ln]
    if loose:
        raise PackagingError(f"requirements.txt must pin every package (==); unpinned: {loose}")
    return lines


def check_doc(path: Path, allow_draft: bool) -> list[str]:
    if not path.exists():
        if allow_draft:
            return [f"methodology doc {path} missing: packaging the blank template (--allow-draft-doc)"]
        raise PackagingError(f"{path} not found: copy data/raw/Documentation_template.md there and fill it in")
    left = [p for p in PLACEHOLDERS if p in path.read_text(encoding="utf-8")]
    if left and not allow_draft:
        raise PackagingError(f"{path} still has template placeholders: {left}")
    return [f"methodology doc still has placeholders {left}"] if left else []


def build_zip(zip_path: Path, out_dir: Path, doc: Path, include_model: bool) -> list[str]:
    files = {f"output/{MATCHING_FILE}": out_dir / MATCHING_FILE, f"output/{CANDIDATE_FILE}": out_dir / CANDIDATE_FILE}
    for p in sorted((ROOT / "src").glob("*.py")) + EXTRA_SRC:
        files[f"{PKG}/src/{p.name}"] = p
    files[f"{PKG}/README.md"] = README
    files[f"{PKG}/requirements.txt"] = ROOT / "requirements.txt"
    files["Documentation_template.md"] = doc if doc.exists() else ROOT / "data" / "raw" / "Documentation_template.md"
    if include_model and MODEL_PATH.exists():
        files[f"{PKG}/outputs/model.joblib"] = MODEL_PATH
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, src in files.items():
            z.write(src, arc)
    return sorted(files)


def self_check(zip_path: Path, expected: list[str]) -> None:
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if not n.endswith("/"))
        if names != expected:
            raise PackagingError(f"zip content mismatch: extra {set(names) - set(expected)}, missing {set(expected) - set(names)}")
        bad = [n for n in names if n.endswith(".tsv") and not n.startswith("output/")
               or any(x in n for x in ("__pycache__", ".venv", "data/", ".git"))]
        if bad:
            raise PackagingError(f"forbidden files in zip: {bad}")
        with tempfile.TemporaryDirectory() as tmp:
            z.extractall(tmp)
            code = Path(tmp) / PKG
            probe = "import importlib; [importlib.import_module('src.' + m) for m in %r]" % MODULES
            r = subprocess.run([sys.executable, "-c", probe], cwd=code, capture_output=True, text=True)
            if r.returncode != 0:
                raise PackagingError("packaged code does not import on its own:\n" + r.stderr[-2000:])


def package(out_dir: Path = OUTPUT_DIR, test_dir: Path = DATA_DIR / "test", doc: Path = DOC,
            zip_path: Path | None = None, check_ids: bool = False, allow_empty: bool = False,
            allow_draft_doc: bool = False, include_model: bool = True, log=print) -> Path:
    zip_path = zip_path or out_dir / f"{TEAM}_submission.zip"
    warnings = []
    log(validate_outputs(out_dir, test_dir, check_ids).strip().splitlines()[-1])
    rows, nonempty = count_nonempty(out_dir / MATCHING_FILE)
    if nonempty == 0 and not allow_empty:
        raise PackagingError(f"all {rows:,} matched lists are empty (looks like a --dry-run output); "
                             "pass --allow-empty only for a packaging test")
    check_requirements(ROOT / "requirements.txt")
    warnings += check_doc(doc, allow_draft_doc)
    if include_model and not MODEL_PATH.exists():
        warnings.append(f"{MODEL_PATH} not found: zip will not include the trained model")
    names = build_zip(zip_path, out_dir, doc, include_model)
    self_check(zip_path, names)
    for w in warnings:
        log(f"WARNING: {w}")
    log(f"OK {zip_path} ({zip_path.stat().st_size / 2**20:.1f} MB): {rows:,} S1 rows, {nonempty:,} with matches; "
        f"{len(names)} files; packaged code imports on its own")
    return zip_path


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # never crash on printing validator text
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-ids", action="store_true", help="validator also checks IDs exist (few GB RAM)")
    ap.add_argument("--allow-empty", action="store_true", help="allow all-empty matches (packaging test only)")
    ap.add_argument("--allow-draft-doc", action="store_true", help="allow a missing/unfinished methodology doc")
    ap.add_argument("--no-model", action="store_true", help="do not include outputs/model.joblib")
    args = ap.parse_args()
    try:
        package(check_ids=args.check_ids, allow_empty=args.allow_empty, allow_draft_doc=args.allow_draft_doc,
                include_model=not args.no_model)
    except PackagingError as e:
        print(f"PACKAGING FAILED: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
