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

SEED = 42
BETA = 0.5  # F-beta used by the competition (macro, per Source 1 entity)
