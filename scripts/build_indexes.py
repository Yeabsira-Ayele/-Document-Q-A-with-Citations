"""Build one Chroma index per chunking strategy.

Usage:
    python scripts/build_indexes.py                 # both strategies
    python scripts/build_indexes.py --only section  # just one
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk, describe, load_documents  # noqa: E402
from src.config import STRATEGIES  # noqa: E402
from src.vectorstore import build_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=STRATEGIES, help="build a single strategy")
    args = parser.parse_args()

    targets = (args.only,) if args.only else STRATEGIES
    pages = load_documents()

    if len(pages) < 50:
        print(f"\nWarning: corpus is {len(pages)} pages. The brief requires at "
              f"least 50 — add more PDFs to Data/.\n")

    for strategy in targets:
        chunks = chunk(strategy, pages)
        print(f"\n[{strategy}] {describe(chunks)}")
        build_index(strategy, chunks)

    print("\nDone. Next: python eval/run_eval.py")


if __name__ == "__main__":
    main()