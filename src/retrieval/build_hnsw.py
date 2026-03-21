"""
Build an HNSW index over paragraph embeddings

Usage:
    python src/retrieval/build_hnsw.py --config configs/build_hnsw.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence

import hnswlib
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from tools import load_yaml, read_access_token


# ---------------------- Config ----------------------

@dataclass
class BuilderConfig:
    dataset_path: str
    dataset_name: str | None
    split: str
    output_dir: Path
    min_paragraph_size: int
    model: str
    batch_size: int
    initial_index_size: int | None
    ef_construction: int
    m: int
    ef_search: int
    num_threads: int
    metadata_filename: str
    access_config: str | None = "configs/access.yaml"

    @classmethod
    def from_file(cls, path: Path) -> "BuilderConfig":
        raw = load_yaml(path)
        required = [
            "dataset_path",
            "dataset_name",
            "split",
            "output_dir",
            "min_paragraph_size",
            "model",
            "batch_size",
            "initial_index_size",
            "ef_construction",
            "m",
            "ef_search",
            "num_threads",
            "metadata_filename",
        ]
        missing = [k for k in required if k not in raw]
        if missing:
            raise KeyError(f"Config missing required keys: {missing}")
        return cls(
            dataset_path=raw["dataset_path"],
            dataset_name=raw.get("dataset_name"),
            split=raw["split"],
            output_dir=Path(raw["output_dir"]),
            min_paragraph_size=raw.get("min_paragraph_size"),
            model=raw["model"],
            batch_size=int(raw["batch_size"]),
            initial_index_size=raw.get("initial_index_size"),
            ef_construction=int(raw["ef_construction"]),
            m=int(raw["m"]),
            ef_search=int(raw["ef_search"]),
            num_threads=int(raw["num_threads"]),
            metadata_filename=raw["metadata_filename"],
            access_config=raw.get("access_config", "configs/access.yaml"),
        )


# ---------------------- Builder ----------------------

class HnswIndexBuilder:
    def __init__(self, cfg: BuilderConfig) -> None:
        self.cfg = cfg
        self.out_dir = cfg.output_dir
        self.meta_path = self.out_dir / cfg.metadata_filename
        self.index_path = self.out_dir / "index.bin"
        self.manifest_path = self.out_dir / "manifest.json"
        self.paragraphs_path = self.out_dir / "paragraphs.jsonl"

        self.access_token = read_access_token(cfg.access_config)
        self.model = SentenceTransformer(
            cfg.model,
            use_auth_token=self.access_token or None,
            model_kwargs={"torch_dtype": torch.float32},
            device="cpu",
        )
        self.dim = self.model.get_sentence_embedding_dimension()

        self.dataset = load_dataset(cfg.dataset_path, cfg.dataset_name, split=cfg.split)

        self.max_elements = cfg.initial_index_size
        self.index = hnswlib.Index(space="ip", dim=self.dim)
        self.index.init_index(
            max_elements=self.max_elements, ef_construction=cfg.ef_construction, M=cfg.m
        )
        self.index.set_ef(cfg.ef_search)
        self.index.set_num_threads(cfg.num_threads)

        self._log_init_summary(cfg)

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split raw article text into paragraphs, keeping only non-empty parts."""
        parts = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > self.cfg.min_paragraph_size]
        return parts or ([text.strip()] if text.strip() else [])

    def build(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        next_paragraph_idx = 0

        with self.meta_path.open("w", encoding="utf-8") as meta_file, \
                self.paragraphs_path.open("w", encoding="utf-8") as paragraph_file:

            text_batch: List[str] = []
            meta_batch: List[dict] = []

            for article_idx, row in enumerate(tqdm(self.dataset, desc="Articles", unit="article")):
                if article_idx > 100:
                    break
                paragraphs = self._split_paragraphs(row["text"])
                for paragraph_idx, paragraph in enumerate(paragraphs):
                    text_batch.append(paragraph)
                    meta_batch.append(
                        {
                            "row_id": row.get("id", ""),
                            "url": row.get("url", ""),
                            "title": row.get("title", ""),
                            "paragraph_index": paragraph_idx,
                        }
                    )

                    if len(text_batch) >= self.cfg.batch_size:
                        next_paragraph_idx = self._encode_batch(text_batch, meta_batch, meta_file, 
                                                                paragraph_file, next_paragraph_idx)
                        text_batch.clear()
                        meta_batch.clear()

            if text_batch:
                next_paragraph_idx = self._encode_batch(text_batch, meta_batch, meta_file, 
                                                        paragraph_file, next_paragraph_idx)

        self.index.save_index(str(self.index_path))
        self._write_manifest(vector_count=next_paragraph_idx)

        self._log_build_summary(next_paragraph_idx)

    def _encode_batch(
        self,
        text_batch: Sequence[str],
        meta_batch: Sequence[dict],
        meta_file,
        paragraph_file,
        start_id: int,
    ) -> int:
        embeddings = self.model.encode_document(
            list(text_batch),
            convert_to_numpy=True,
            batch_size=len(text_batch),
            show_progress_bar=False,
            normalize_embeddings=False,
        )

        needed_index_capacity = start_id + len(embeddings)
        if needed_index_capacity > self.index.get_max_elements():
            new_index_capacity = int(math.ceil(needed_index_capacity * 1.2))
            self.index.resize_index(new_index_capacity)
            logging.info("Resized index to %d elements", new_index_capacity)

        ids = np.arange(start_id, start_id + len(embeddings))
        self.index.add_items(embeddings, ids)

        for meta, para_text, pid in zip(meta_batch, text_batch, ids):
            meta_file.write(json.dumps(meta, ensure_ascii=False) + "\n")
            paragraph_file.write(json.dumps({"pid": int(pid), "text": para_text}, ensure_ascii=False) + "\n")

        return start_id + len(embeddings)

    def _write_manifest(self, vector_count: int) -> None:
        manifest = {
            "dataset": self.cfg.dataset_path,
            "dataset_name": self.cfg.dataset_name,
            "split": self.cfg.split,
            "model": self.cfg.model,
            "dim": self.dim,
            "metric": "ip",
            "ef_search": self.cfg.ef_search,
            "ef_construction": self.cfg.ef_construction,
            "M": self.cfg.m,
            "metadata_file": self.meta_path.name,
            "hnsw_index_file": self.index_path.name,
            "paragraphs_file": self.paragraphs_path.name,
            "vector_count": vector_count
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
    def _log_init_summary(self, cfg: BuilderConfig) -> None:
        if self.access_token:
            logging.info("Using access token from %s", cfg.access_config)
        logging.info(
            "Loaded dataset '%s' (name=%s, split=%s) rows=%d",
            cfg.dataset_path,
            cfg.dataset_name,
            cfg.split,
            len(self.dataset),
        )
        logging.info("Loaded encoder '%s' with dim=%d", cfg.model, self.dim)
        logging.info(
            "Initialized HNSW (ip): M=%d ef_construction=%d ef_search=%d max_elements=%d",
            cfg.m,
            cfg.ef_construction,
            cfg.ef_search,
            self.max_elements,
        )

    def _log_build_summary(self, vectors: int) -> None:
        logging.info("Indexed %d paragraphs", vectors)
        logging.info("Index saved to %s", self.index_path)
        logging.info("Metadata saved to %s", self.meta_path)


# ---------------------- CLI ----------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HNSW index over wiki paragraphs.")
    parser.add_argument(
        "--config",
        default="configs/build_hnsw.yaml",
        help="Path to JSON or YAML config. Defaults to configs/build_hnsw.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = BuilderConfig.from_file(Path(args.config))
    builder = HnswIndexBuilder(cfg)
    builder.build()


if __name__ == "__main__":
    main()
