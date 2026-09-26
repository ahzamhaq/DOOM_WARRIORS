"""End-to-end test inference -> outputs/matching_results.tsv + candidate_pairs.tsv.

Run from repo root:  python -m src.predict
Owner: Lead (integration).
"""
from src.data_loader import load_source, write_submission


def main():
    s1 = load_source("test", 1)
    s1_ids = s1["entity_id"].tolist()
    # TODO: normalize -> blocking -> features -> matcher
    matches, candidates = {}, {}
    write_submission(matches, candidates, s1_ids)


if __name__ == "__main__":
    main()
