"""Central paths and constants. Import from here instead of hard-coding paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "dataset"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

MATCHING_FILE = "matching_results.tsv"
CANDIDATE_FILE = "candidate_pairs.tsv"

# ---- Integration contract (see docs/CONTRACT.md) ----
S1_ID, CAND_ID, PROBA = "s1_id", "cand_id", "proba"
NORM_COLUMNS = ["name_norm", "addr_norm", "name_is_domain", "name_script", "addr_empty", "house_no"]
BLK_COLUMNS = ["blk_name_score", "blk_addr_score", "blk_name_rank", "blk_addr_rank"]
CHUNK_S1 = 50_000          # S1 entities per processing chunk
BLOCK_K = 20               # candidates per retriever per S1
MAX_CANDS_PER_S1 = 40      # hard cap after unioning retrievers
PROBA_FLOOR = 0.05         # scored pairs below this are dropped before resolution
MODEL_PATH = OUTPUT_DIR / "model.joblib"
# EXPERIMENTAL: each S2/S3 record matched at most one S1 in train. Off until Person 3 shows on validation
# that enforcing it helps (docs/CONTRACT.md §6). Flip here, not inside the matcher.
ONE_TO_ONE = False

# ---- Frozen dev world (scripts/build_dev_world.py, docs/CONTRACT.md §11) ----
DEV_WORLD_DIR = PROCESSED_DIR / "dev_world"
DEV_MANIFEST = ROOT / "docs" / "dev_world_manifest.json"
DEV_N_S1 = 30_000
DEV_VAL_FRAC = 0.2

SEED = 42
BETA = 0.5  # F-beta used by the competition (macro, per Source 1 entity)
