"""Extract features for every clip in manifest.jsonl -> features.jsonl.

Usage: python scripts/extract_features.py <bench_dir>
Resume-safe: skips clip_ids already present in features.jsonl.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from synsync.features import extract_benchmark  # noqa: E402


def main() -> None:
    n = extract_benchmark(sys.argv[1])
    print(f"done: {n} new feature rows")


if __name__ == "__main__":
    main()
