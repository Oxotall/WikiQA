"""
Build an HNSW index over paragraph embeddings.

Usage:
    python src/retrieval/build_hnsw.py --config configs/build_hnsw.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hnsw_index import HnswIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HNSW index over wiki paragraphs.")
    parser.add_argument(
        "--config",
        default="configs/build_hnsw.yaml",
        help="Path to YAML config. Defaults to configs/build_hnsw.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    HnswIndex.build_from_config(Path(args.config))


if __name__ == "__main__":
    main()
