"""
Flask RAG service backed by the prebuilt HNSW index.

- Loads search/generation parameters from configs/serve.yaml
- Serves a single-page UI that shows the answer and supporting paragraphs.

Run:
    FLASK_APP=src/api/app.py flask run --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.retrieval.hnsw_index import HnswIndex
from src.retrieval.tools import load_yaml, read_access_token


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = Path("configs/serve.yaml")


@dataclass
class ServiceConfig:
    index_dir: Path
    encoder_model: str
    qa_model: str
    model_device: str
    k: int
    max_new_tokens: int
    access_config: str | None

    @classmethod
    def from_file(cls, path: Path) -> "ServiceConfig":
        cfg = load_yaml(path)
        return cls(
            index_dir=Path(cfg["index_dir"]),
            encoder_model=cfg.get("encoder_model"),
            qa_model=cfg.get("qa_model"),
            model_device=cfg.get("model_device"),
            k=int(cfg.get("k", 5)),
            max_new_tokens=int(cfg.get("max_new_tokens", 128)),
            access_config=cfg.get("access_config"),
        )


# ------------------------- Components -------------------------

class RagIndex:
    def __init__(self, cfg: ServiceConfig, access_token: str | None):
        manifest_path = cfg.index_dir / "manifest.json"
        self.hnsw_index = HnswIndex.load_from_disk(
            manifest_path,
            model_device=cfg.model_device,
        )

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        return self.hnsw_index.search_by_text(query, k=k)


class AnswerGenerator:
    def __init__(self, model_name: str, model_device: str, access_token: str | None):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token or None)
        model_kwargs = {"torch_dtype": "auto", "device_map": model_device}
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
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
    generator = AnswerGenerator(cfg.qa_model, cfg.model_device, access_token)

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
