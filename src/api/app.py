"""
Flask RAG service backed by the prebuilt HNSW index.

- Loads search/generation parameters from configs/serve.yaml (or RAG_CONFIG env).
- Serves a single-page UI that shows the answer and supporting paragraphs.

Run:
    FLASK_APP=src/api/app.py flask run --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import hnswlib
import json
import numpy as np
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.api.tools import load_yaml_or_json, read_access_token


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = Path(os.environ.get("RAG_CONFIG", "configs/serve.yaml"))


@dataclass
class ServiceConfig:
    index_dir: Path
    encoder_model: str
    qa_model: str
    k: int
    max_new_tokens: int
    access_config: str | None

    @classmethod
    def from_file(cls, path: Path) -> "ServiceConfig":
        cfg = load_yaml_or_json(path)
        return cls(
            index_dir=Path(cfg["index_dir"]),
            encoder_model=cfg.get("encoder_model"),
            qa_model=cfg.get("qa_model", "Qwen/Qwen2-1.5B-Instruct"),
            k=int(cfg.get("k", 5)),
            max_new_tokens=int(cfg.get("max_new_tokens", 128)),
            access_config=cfg.get("access_config"),
        )


# ------------------------- Components -------------------------

class RagIndex:
    def __init__(self, cfg: ServiceConfig, access_token: str | None):
        self.index_dir = cfg.index_dir
        manifest_path = self.index_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text())

        meta_path = self.index_dir / self.manifest["metadata_file"]
        para_path = self.index_dir / self.manifest["paragraphs_file"]
        index_path = self.index_dir / self.manifest["hnsw_index_file"]

        self.encoder = SentenceTransformer(
            cfg.encoder_model or self.manifest["model"], use_auth_token=access_token or None
        )

        self.index = hnswlib.Index(space=self.manifest["metric"], dim=self.manifest["dim"])
        self.index.load_index(str(index_path))
        self.index.set_ef(self.manifest.get("ef_search", 64))

        self.metadata = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
        self.paragraphs = [json.loads(line)["text"] for line in para_path.read_text().splitlines() if line.strip()]

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        vec = self.encoder.encode_query([query], convert_to_numpy=True, normalize_embeddings=False)[0]
        labels, dists = self.index.knn_query(vec, k=k)
        rows = []
        for idx, dist in zip(labels[0], dists[0]):
            meta = self.metadata[idx]
            rows.append(
                {
                    "score": float(1 - dist),
                    "title": meta.get("title", ""),
                    "url": meta.get("url", ""),
                    "paragraph": self.paragraphs[idx],
                }
            )
        return rows


class AnswerGenerator:
    def __init__(self, model_name: str, access_token: str | None):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token or None)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto")
        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def build_prompt(self, question: str, rows: List[Dict[str, Any]]) -> str:
        context = "\n\n".join(
            [f"Title: {r['title']}\nURL: {r['url']}\nParagraph: {r['paragraph']}" for r in rows]
        )
        return (
            "You are a concise assistant. Use only the provided context to answer.\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

    def generate(self, question: str, rows: List[Dict[str, Any]], max_new_tokens: int) -> str:
        prompt = self.build_prompt(question, rows)
        out = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return out[0]["generated_text"].split("Answer:", 1)[-1].strip()


# ------------------------- App factory -------------------------

def create_app(config_path: Path = DEFAULT_CONFIG) -> Flask:
    cfg = ServiceConfig.from_file(config_path)
    access_token = read_access_token(cfg.access_config)

    rag_index = RagIndex(cfg, access_token)
    generator = AnswerGenerator(cfg.qa_model, access_token)

    app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

    @app.route("/", methods=["GET", "POST"])
    def home():
        q = ""
        answer = None
        rows: List[Dict[str, Any]] = []
        if request.method == "POST":
            q = request.form.get("q", "").strip()
            if q:
                rows = rag_index.search(q, k=cfg.k)
                answer = generator.generate(q, rows, max_new_tokens=cfg.max_new_tokens)
        return render_template("index.html", q=q, answer=answer, rows=rows)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
