"""
HNSW index
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import hnswlib
import numpy as np
import torch
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

try:
    from src.retrieval.tools import load_yaml, read_access_token
except ImportError:
    from tools import load_yaml, read_access_token


class HnswIndex:
    def __init__(
        self,
        manifest: Mapping[str, Any],
        index: hnswlib.Index,
        sqlite_connection: sqlite3.Connection,
        model: SentenceTransformer,
    ):
        self.manifest = dict(manifest)
        self.index = index
        self.sqlite_connection = sqlite_connection
        self.model = model

    @classmethod
    def build_from_config(cls, config_path: Path | str) -> "HnswIndex":
        cfg = load_yaml(Path(config_path))
        access_token = cls._get_access_token(cfg)
        out_path_dict = cls._prepare_output_paths(cfg)
        model, dim = cls._load_model(cfg, access_token)
        index = cls._init_index(cfg, dim)
        sqlite_connection = cls._open_sqlite(out_path_dict["sqlite_path"])
        cls._init_sqlite_schema(sqlite_connection)
        manifest = cls._create_manifest(cfg, dim, out_path_dict)
        hnsw_index = cls(manifest, index, sqlite_connection, model)

        dataset = cls._load_dataset(cfg)
        hnsw_index._build_paragraph_index(cfg, dataset, out_path_dict["index_path"], 
                                          out_path_dict["manifest_path"], out_path_dict["sqlite_path"])
        
        return hnsw_index

    @classmethod
    def load_from_disk(
        cls,
        manifest_path: Path | str,
        model_device: str = "cpu"
    ) -> "HnswIndex":
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        base_dir = manifest_path.parent

        index_path = base_dir / manifest["hnsw_index_file"]
        sqlite_path = base_dir / manifest["sqlite_file"]

        index = hnswlib.Index(space=manifest["metric"], dim=manifest["dim"])
        index.load_index(str(index_path))
        index.set_ef(manifest.get("ef_search"))

        model = SentenceTransformer(manifest["model"], device=model_device)
        sqlite_connection = cls._open_sqlite(sqlite_path)

        return cls(
            manifest=manifest,
            index=index,
            sqlite_connection=sqlite_connection,
            model=model,
        )

    @staticmethod
    def _get_access_token(cfg: Mapping[str, Any]) -> str | None:
        return read_access_token(cfg.get("access_config", "configs/access.yaml"))

    @staticmethod
    def _prepare_output_paths(cfg: Mapping[str, Any]) -> Dict[str, Path]:
        out_dir = Path(cfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "out_dir": out_dir,
            "index_path": out_dir / "index.bin",
            "manifest_path": out_dir / "manifest.json",
            "sqlite_path": out_dir / "passages.sqlite3",
        }

    @staticmethod
    def _open_sqlite(sqlite_path: Path) -> sqlite3.Connection:
        return sqlite3.connect(sqlite_path, check_same_thread=False)

    @staticmethod
    def _init_sqlite_schema(sqlite_connection: sqlite3.Connection) -> None:
        sqlite_connection.execute(
            """
            CREATE TABLE IF NOT EXISTS passages (
                pid INTEGER PRIMARY KEY,
                row_id TEXT,
                url TEXT,
                title TEXT,
                paragraph_index INTEGER,
                paragraph TEXT
            )
            """
        )
        sqlite_connection.execute("CREATE INDEX IF NOT EXISTS idx_passages_pid ON passages(pid)")
        sqlite_connection.commit()

    @staticmethod
    def _load_model(cfg: Mapping[str, Any], access_token: str | None) -> tuple[SentenceTransformer, int]:
        model_device = cfg.get("model_device", "cpu")
        model = SentenceTransformer(
            cfg["model"],
            use_auth_token=access_token or None,
            model_kwargs={"torch_dtype": torch.float32},
            device=model_device,
        )
        dim = model.get_sentence_embedding_dimension()
        return model, dim

    @staticmethod
    def _load_dataset(cfg: Mapping[str, Any]):
        dataset = load_dataset(cfg["dataset_path"], cfg.get("dataset_name"), split=cfg["split"])
        logging.info(
            "Loaded dataset '%s' (name=%s, split=%s) rows=%d",
            cfg["dataset_path"],
            cfg.get("dataset_name"),
            cfg["split"],
            len(dataset),
        )
        return dataset
    
    @staticmethod
    def _create_manifest(cfg: Mapping[str, Any], dim:int, path_dict: Mapping[str, Path]) -> Dict[str, Any]:
        manifest = {
            "dataset": cfg["dataset_path"],
            "dataset_name": cfg.get("dataset_name"),
            "split": cfg["split"],
            "model": cfg["model"],
            "dim": dim,
            "metric": "ip",
            "ef_search": cfg["ef_search"],
            "ef_construction": cfg["ef_construction"],
            "M": cfg["m"],
            "hnsw_index_file": path_dict["index_path"].name,
            "sqlite_file": path_dict["sqlite_path"].name,
            "vector_count": 0
        }
        return manifest

    @staticmethod
    def _init_index(cfg: Mapping[str, Any], dim: int) -> hnswlib.Index:
        max_elements = cfg.get("initial_index_size")

        index = hnswlib.Index(space="ip", dim=dim)
        index.init_index(
            max_elements=max_elements,
            ef_construction=cfg["ef_construction"],
            M=cfg["m"],
        )
        index.set_ef(cfg["ef_search"])
        index.set_num_threads(cfg["num_threads"])
        return index

    def _build_paragraph_index(
        self,
        cfg: Mapping[str, Any],
        dataset: Dataset,
        index_path: Path,
        manifest_path: Path,
        sqlite_path: Path
    ) -> None:
        text_batch: List[str] = []
        meta_batch: List[dict] = []

        for row_idx, row in enumerate(tqdm(dataset, desc="Articles", unit="article")):
            if cfg["max_articles_to_process"] and row_idx > cfg["max_articles_to_process"]:
                break
            paragraphs = self._split_paragraphs(row["text"], cfg["min_paragraph_size"], 
                                                cfg.get["max_paragraph_size"])
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

                if len(text_batch) >= cfg["batch_size"]:
                    self._process_batch(text_batch, meta_batch)
                    text_batch.clear()
                    meta_batch.clear()

        if text_batch:
            self._process_batch(text_batch, meta_batch)
        
        self._save_data_to_disk(index_path, manifest_path)
        self._log_build_summary(index_path, sqlite_path)
    
    def _process_batch(self, text_batch: Sequence[str], meta_batch: Sequence[dict]) -> None:
        embeddings = self.model.encode_document(
            list(text_batch),
            convert_to_numpy=True,
            batch_size=len(text_batch),
            show_progress_bar=False,
            normalize_embeddings=False,
        )

        needed_index_capacity = self.manifest['vector_count'] + len(embeddings)
        if needed_index_capacity > self.index.get_max_elements():
            new_index_capacity = int(math.ceil(needed_index_capacity * 1.2))
            self.index.resize_index(new_index_capacity)
            logging.info("Resized index to %d elements", new_index_capacity)

        ids = np.arange(self.manifest['vector_count'], self.manifest['vector_count'] + len(embeddings))
        self.index.add_items(embeddings, ids)

        for meta, paragraph_text, pid in zip(meta_batch, text_batch, ids):
            self.sqlite_connection.execute(
                """
                INSERT INTO passages (pid, row_id, url, title, paragraph_index, paragraph)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(pid),
                    meta.get("row_id", ""),
                    meta.get("url", ""),
                    meta.get("title", ""),
                    int(meta.get("paragraph_index", -1)),
                    paragraph_text,
                ),
            )

        self.manifest['vector_count'] += len(embeddings)
    
    @staticmethod
    def _split_paragraphs(
        text: str,
        min_paragraph_size: int,
        max_paragraph_size: int | None = None,
    ) -> List[str]:
        split_parts: List[str] = []
        for paragraph in re.split(r"\n{2,}", text):
            if len (paragraph) < min_paragraph_size:
                continue
            if len(paragraph) <= max_paragraph_size:
                split_parts.append(paragraph)
                continue
            split_parts.extend(
                paragraph[i : i + max_paragraph_size]
                for i in range(0, len(paragraph), max_paragraph_size)
            )
        return split_parts

    def _save_data_to_disk(self, index_path, manifest_path) -> None:
        self.sqlite_connection.commit()
        self.index.save_index(str(index_path))
        manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    def _log_build_summary(self, index_path: Path, sqlite_path: Path) -> None:
        logging.info("Indexed %d paragraphs", self.manifest['vector_count'])
        logging.info("Index saved to %s", index_path)
        logging.info("SQLite saved to %s", sqlite_path)

# -------------------------- Search -------------------------- 

    def _fetch_passages_by_ids(self, ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        cursor = self.sqlite_connection.execute(
            f"""
            SELECT pid, row_id, url, title, paragraph_index, paragraph
            FROM passages
            WHERE pid IN ({placeholders})
            """,
            tuple(ids),
        )
        rows = {}
        for pid, row_id, url, title, paragraph_index, paragraph in cursor.fetchall():
            rows[int(pid)] = {
                "row_id": row_id,
                "url": url,
                "title": title,
                "paragraph_index": paragraph_index,
                "paragraph": paragraph,
            }
        return rows

    def search_by_vector(self, vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
        labels, distances = self.index.knn_query(vector, k=k)
        ids = [int(idx) for idx in labels[0]]
        rows_by_id = self._fetch_passages_by_ids(ids)
        rows = []
        for idx, dist in zip(ids, distances[0]):
            passage = rows_by_id.get(idx, {})
            rows.append(
                {
                    "score": float(1 - dist),
                    "title": passage.get("title", ""),
                    "url": passage.get("url", ""),
                    "paragraph": passage.get("paragraph", ""),
                }
            )
        return rows

    def search_by_text(self, query: str, k: int) -> List[Dict[str, Any]]:
        vector = self.model.encode_query([query], convert_to_numpy=True, normalize_embeddings=False)[0]
        return self.search_by_vector(vector, k)
