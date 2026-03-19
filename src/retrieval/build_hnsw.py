"""
Build an HNSW index over paragraph-level embeddings from the wiki dataset.

Config-driven script:
1) Loads the dataset split (expects fields: id, url, title, text)
2) Splits each article text into paragraphs
3) Embeds paragraphs with a SentenceTransformer model
4) Adds vectors to an hnswlib index and writes matching metadata rows

Run:
    python src/retrieval/build_hnsw.py --config configs/build_hnsw.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import torch
import logging
import yaml
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import hnswlib
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


def split_paragraphs(text: str) -> List[str]:
    """Split raw article text into paragraphs, keeping only non-empty parts."""
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HNSW index over wiki paragraphs.")
    parser.add_argument( 
        "--config",
        default="configs/build_hnsw.yaml",
        help="Path to JSON or YAML config containing required params. Defaults to configs/build_hnsw.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Starting HNSW build using config %s", args.config)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / cfg["metadata_filename"]
    index_path = out_dir / "index.bin"
    manifest_path = out_dir / "manifest.json"

    # Load dataset; keeping it non-streaming to allow optional length checks.
    dataset = load_dataset(cfg["dataset_path"], cfg["dataset_name"], split=cfg["split"])
    logging.info("Loaded dataset '%s' (name=%s, split=%s) with %d rows", cfg["dataset_path"], cfg["dataset_name"], cfg["split"], len(dataset))

    access_token = read_access_token(cfg.get("access_config", "configs/access.yaml"))
    if access_token:
        logging.info("Using access token from %s", cfg.get("access_config", "configs/access.yaml"))
    model = SentenceTransformer(cfg["model"], use_auth_token=access_token or None, model_kwargs={"torch_dtype": torch.float32}, device='cpu')
    dim = model.get_sentence_embedding_dimension()
    logging.info("Loaded encoder '%s' with dimension %d", cfg["model"], dim)

    # First pass: count total paragraphs to size index and storage.
    logging.info("Counting paragraphs...")
    total_paragraphs = 0
    for row in tqdm(dataset, desc="Count paragraphs", unit="article"):
        total_paragraphs += len(split_paragraphs(row["text"]))
    logging.info("Total paragraphs: %d", total_paragraphs)

    # Initialize index and storage sized to paragraph count (or configured max).
    max_elements = cfg["max_elements"] or total_paragraphs
    index = hnswlib.Index(space="ip", dim=dim)
    index.init_index(max_elements=max_elements, ef_construction=cfg["ef_construction"], M=cfg["m"])
    index.set_ef(cfg["ef_search"])
    index.set_num_threads(cfg["num_threads"])
    logging.info("Initialized HNSW: M=%d, ef_construction=%d, ef_search=%d, max_elements=%d", cfg["m"], cfg["ef_construction"], cfg["ef_search"], max_elements)

    # Memory-map for embeddings and file for paragraphs (aligned by vector id).
    embeddings_path = out_dir / "embeddings.dat"
    embeddings_mm = np.memmap(embeddings_path, dtype="float32", mode="w+", shape=(max_elements, dim))
    paragraphs_path = out_dir / "paragraphs.jsonl"

    next_id = 0
    written_meta = 0

    with meta_path.open("w", encoding="utf-8") as meta_file, paragraphs_path.open("w", encoding="utf-8") as para_file:
        # Iterate articles and accumulate paragraph batches for embedding.
        text_batch: List[str] = []
        meta_batch: List[dict] = []

        for article_idx, row in enumerate(tqdm(dataset, desc="Articles", unit="article")):
            paragraphs = split_paragraphs(row["text"])
            if article_idx > 100:
                break
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

                if len(text_batch) >= cfg["batch_size"]:
                    next_id = _flush_batch(
                        text_batch,
                        meta_batch,
                        model,
                        index,
                        meta_file,
                        para_file,
                        embeddings_mm,
                        next_id,
                    )
                    written_meta += len(meta_batch)
                    text_batch.clear()
                    meta_batch.clear()


        # Flush remaining paragraphs.
        if text_batch:
            next_id = _flush_batch(text_batch, meta_batch, model, index, meta_file, para_file, embeddings_mm, next_id)
            written_meta += len(meta_batch)

    # Ensure memmap is flushed.
    embeddings_mm.flush()
    # Report any NaN vectors.
    nan_rows = int(np.isnan(embeddings_mm).any(axis=1).sum())
    if nan_rows:
        logging.warning("Found %d vectors containing NaNs in embeddings memmap", nan_rows)
    else:
        logging.info("No NaNs detected in stored embeddings")

    # Persist index and a small manifest for downstream usage.
    index.save_index(str(index_path))
    manifest = {
        "dataset": cfg["dataset_path"],
        "dataset_name": cfg["dataset_name"],
        "split": cfg["split"],
        "model": cfg["model"],
        "dim": dim,
        "metric": "ip",
        "ef_search": cfg["ef_search"],
        "ef_construction": cfg["ef_construction"],
        "M": cfg["m"],
        "metadata_file": str(meta_path.name),
        "hnsw_index_file": str(index_path.name),
        "embeddings_file": str(embeddings_path.name),
        "paragraphs_file": str(paragraphs_path.name),
        "vectors": next_id,
        "paragraphs": next_id,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logging.info("Indexed %d paragraphs; metadata rows written: %d", next_id, written_meta)
    logging.info("Index saved to %s", index_path)
    logging.info("Metadata saved to %s", meta_path)


def _flush_batch(
    text_batch: Sequence[str],
    meta_batch: Sequence[dict],
    model: SentenceTransformer,
    index: hnswlib.Index,
    meta_file,
    para_file,
    embeddings_mm: np.memmap,
    start_id: int,
) -> int:
    """Embed and add a batch to the index, writing matching metadata."""
    embeddings = model.encode_document(
        list(text_batch),
        convert_to_numpy=True,
        batch_size=len(text_batch),
        show_progress_bar=False,
        normalize_embeddings=False,
    )

    # Resize index if needed (hnswlib allows growth via resize_index).
    needed = start_id + len(embeddings)
    if needed > index.get_max_elements():
        new_cap = int(math.ceil(needed * 1.2))
        index.resize_index(new_cap)

    ids = np.arange(start_id, start_id + len(embeddings))
    index.add_items(embeddings, ids)

    # Persist metadata, paragraphs, and embeddings aligned by id.
    for meta, para_text, emb_row, pid in zip(meta_batch, text_batch, embeddings, ids):
        meta_file.write(json.dumps(meta, ensure_ascii=False) + "\n")
        para_file.write(json.dumps({"pid": int(pid), "text": para_text}, ensure_ascii=False) + "\n")
        embeddings_mm[int(pid)] = emb_row

    return start_id + len(embeddings)


def load_config(path: str) -> Mapping[str, object]:
    """Load config from JSON or YAML file and validate required keys."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(cfg_path.read_text())
    else:
        raw = json.loads(cfg_path.read_text())

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
    return raw


def read_access_token(path: str) -> str | None:
    """Read an access token from a YAML config.

    Expected format (configs/access.yaml):
        access_token: hf_xxx
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        return None
    data = yaml.safe_load(cfg_path.read_text())
    if isinstance(data, dict) and "access_token" in data and data["access_token"]:
        return str(data["access_token"]).strip()
    return None


if __name__ == "__main__":
    main()
