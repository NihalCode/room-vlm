"""YAML configuration loading and structured access."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def load_config(path: str | Path, base: str | Path | None = None) -> dict[str, Any]:
    """Load a config YAML, optionally merging over a base config."""
    cfg = load_yaml(path)
    if base is not None:
        cfg = _deep_merge(load_yaml(base), cfg)
    elif Path(path).name != "base.yaml":
        base_candidate = Path(path).parent / "base.yaml"
        if base_candidate.exists() and Path(path).resolve() != base_candidate.resolve():
            cfg = _deep_merge(load_yaml(base_candidate), cfg)
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


@dataclass
class AppConfig:
    """Thin wrapper around the nested config dict."""

    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path, base: str | Path | None = None) -> "AppConfig":
        return cls(load_config(path, base=base))

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def seed(self) -> int:
        return int(self.get("project", "seed", default=42))

    @property
    def labels(self) -> list[str]:
        return list(self.get("labels", default=[]))

    @property
    def model_name(self) -> str:
        return str(self.get("model", "name", default="Qwen/Qwen3-VL-4B-Instruct"))
