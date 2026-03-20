"""Shared utilities for retrieval and serving components."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Mapping

import yaml


def load_yaml_or_json(path: Path) -> Mapping[str, Any]:
    """Load a YAML or JSON file into a dict."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def read_access_token(path: str | None) -> str | None:
    """Read HF access token from YAML file with key 'access_token'."""
    if not path:
        return None
    cfg_path = Path(path)
    if not cfg_path.exists():
        return None
    data = yaml.safe_load(cfg_path.read_text())
    if isinstance(data, dict):
        token = data.get("access_token")
        if token:
            return str(token).strip()
    return None

