"""
Build an HNSW index over paragraph embeddings in a class-oriented, config-driven way.

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
class BuildConfig:
    dataset_path: str
    dataset_name: str | None
    split: str
    output_dir: Path
    model: str
    batch_size: int
    max_elements: int | None
    ef_construction: int
    m: int
    ef_search: int
    num_threads: int
    metadata_filename: str
    access_config: str | None = "configs/access.yaml"

    @classmethod
    def from_file(cls, path: Path) -> "BuildConfig":
        raw = load_yaml(path)
        required = [
            "dataset_path",
            "dataset_name",
            "split",
            "output_dir",
            "model",
            "batch_size",
            "max_elements",
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
            model=raw["model"],
            batch_size=int(raw["batch_size"]),
            max_elements=raw.get("max_elements"),
            ef_construction=int(raw["ef_construction"]),
            m=int(raw["m"]),
            ef_search=int(raw["ef_search"]),
            num_threads=int(raw["num_threads"]),
            metadata_filename=raw["metadata_filename"],
            access_config=raw.get("access_config", "configs/access.yaml"),
        )


# ---------------------- Builder ----------------------

class HnswBuilder:
    def __init__(self, cfg: BuildConfig) -> None:
        self.cfg = cfg
        self.out_dir = cfg.output_dir
        self.meta_path = self.out_dir / cfg.metadata_filename
        self.index_path = self.out_dir / "index.bin"
        self.manifest_path = self.out_dir / "manifest.json"
        self.paragraphs_path = self.out_dir / "paragraphs.jsonl"
        self.embeddings_path = self.out_dir / "embeddings.dat"

        self.access_token = read_access_token(cfg.access_config)
        if self.access_token:
            logging.info("Using access token from %s", cfg.access_config)

        self.model = SentenceTransformer(
            cfg.model,
            use_auth_token=self.access_token or None,
            model_kwargs={"torch_dtype": torch.float32},
            device="cpu",
        )
        self.dim = self.model.get_sentence_embedding_dimension()
        logging.info("Loaded encoder '%s' with dim=%d", cfg.model, self.dim)

        self.dataset = load_dataset(cfg.dataset_path, cfg.dataset_name, split=cfg.split)
        logging.info(
            "Loaded dataset '%s' (name=%s, split=%s) rows=%d",
            cfg.dataset_path,
            cfg.dataset_name,
            cfg.split,
            len(self.dataset),
        )

        self.total_paragraphs = self._count_paragraphs()
        self.max_elements = cfg.max_elements or self.total_paragraphs

        self.index = hnswlib.Index(space="ip", dim=self.dim)
        self.index.init_index(
            max_elements=self.max_elements, ef_construction=cfg.ef_construction, M=cfg.m
        )
        self.index.set_ef(cfg.ef_search)
        self.index.set_num_threads(cfg.num_threads)
        logging.info(
            "Initialized HNSW (ip): M=%d ef_construction=%d ef_search=%d max_elements=%d",
            cfg.m,
            cfg.ef_construction,
            cfg.ef_search,
            self.max_elements,
        )

        self.embeddings_mm = np.memmap(
            self.embeddings_path, dtype="float32", mode="w+", shape=(self.max_elements, self.dim)
        )

    def _count_paragraphs(self) -> int:
        logging.info("Counting paragraphs...")
        total = 0
        for row in tqdm(self.dataset, desc="Count paragraphs", unit="article"):
            total += len(self._split_paragraphs(row["text"]))
        logging.info("Total paragraphs: %d", total)
        return total

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split raw article text into paragraphs, keeping only non-empty parts."""
        parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        return parts or ([text.strip()] if text.strip() else [])

    def build(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        next_id = 0
        written_meta = 0

        with self.meta_path.open("w", encoding="utf-8") as meta_file, \
                self.paragraphs_path.open("w", encoding="utf-8") as para_file:

            text_batch: List[str] = []
            meta_batch: List[dict] = []

            for article_idx, row in enumerate(tqdm(self.dataset, desc="Articles", unit="article")):
                if article_idx > 100:
                    break
                paragraphs = self._split_paragraphs(row["text"])
                for para_idx, para in enumerate(paragraphs):
                    text_batch.append(para)
                    meta_batch.append(
                        {
                            "row_id": row.get("id", ""),
                            "url": row.get("url", ""),
                            "title": row.get("title", ""),
                            "paragraph_index": para_idx,
                        }
                    )

                    if len(text_batch) >= self.cfg.batch_size:
                        next_id = self._encode_batch(text_batch, meta_batch, meta_file, para_file, next_id)
                        written_meta += len(meta_batch)
                        text_batch.clear()
                        meta_batch.clear()

            if text_batch:
                next_id = self._encode_batch(text_batch, meta_batch, meta_file, para_file, next_id)
                written_meta += len(meta_batch)

        self.embeddings_mm.flush()
        nan_rows = int(np.isnan(self.embeddings_mm).any(axis=1).sum())
        if nan_rows:
            logging.warning("Found %d vectors containing NaNs in embeddings memmap", nan_rows)
        else:
            logging.info("No NaNs detected in stored embeddings")

        self.index.save_index(str(self.index_path))
        self._write_manifest(vectors=next_id)

        logging.info("Indexed %d paragraphs; metadata rows written: %d", next_id, written_meta)
        logging.info("Index saved to %s", self.index_path)
        logging.info("Metadata saved to %s", self.meta_path)

    def _encode_batch(
        self,
        text_batch: Sequence[str],
        meta_batch: Sequence[dict],
        meta_file,
        para_file,
        start_id: int,
    ) -> int:
        embeddings = self.model.encode_document(
            list(text_batch),
            convert_to_numpy=True,
            batch_size=len(text_batch),
            show_progress_bar=False,
            normalize_embeddings=False,
        )

        needed = start_id + len(embeddings)
        if needed > self.index.get_max_elements():
            new_cap = int(math.ceil(needed * 1.2))
            self.index.resize_index(new_cap)
            logging.info("Resized index to %d elements", new_cap)

        ids = np.arange(start_id, start_id + len(embeddings))
        self.index.add_items(embeddings, ids)

        for meta, para_text, emb_row, pid in zip(meta_batch, text_batch, embeddings, ids):
            meta_file.write(json.dumps(meta, ensure_ascii=False) + "\n")
            para_file.write(json.dumps({"pid": int(pid), "text": para_text}, ensure_ascii=False) + "\n")
            self.embeddings_mm[int(pid)] = emb_row

        return start_id + len(embeddings)

    def _write_manifest(self, vectors: int) -> None:
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
            "embeddings_file": self.embeddings_path.name,
            "paragraphs_file": self.paragraphs_path.name,
            "vectors": vectors,
            "paragraphs": vectors,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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
    cfg = BuildConfig.from_file(Path(args.config))
    builder = HnswBuilder(cfg)
    builder.build()


if __name__ == "__main__":
    main()
