from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_name: str
    temperature: float
    max_output_tokens: int


@dataclass(frozen=True)
class QualityConfig:
    min_score: float
    require_checker_pass: bool
    max_refinement_attempts: int


@dataclass(frozen=True)
class AppConfig:
    num_problems: int
    max_workers: int
    output_dir: str
    default_batch_id: str
    engineer_model: ModelConfig
    checker_model: ModelConfig
    target_model: ModelConfig
    quality: QualityConfig


def _model(raw: dict[str, Any], env_name: str) -> ModelConfig:
    # Keep Gemini 3.1 Pro Preview as the application default. An environment
    # override is supported for advanced users, but an empty override is ignored.
    model = (os.getenv(env_name) or raw.get("model_name") or DEFAULT_GEMINI_MODEL).strip()
    return ModelConfig(
        provider=str(raw.get("provider", "gemini")).lower(),
        model_name=model,
        temperature=float(raw.get("temperature", 0.4)),
        max_output_tokens=int(raw.get("max_output_tokens", 2048)),
    )


def load_config(path: str | Path = "config/settings.yaml") -> AppConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AppConfig(
        num_problems=int(data.get("num_problems", 10)),
        max_workers=max(1, int(data.get("max_workers", 5))),
        output_dir=str(data.get("output_dir", "./results")),
        default_batch_id=str(data.get("default_batch_id", "batch_01")),
        engineer_model=_model(data.get("engineer_model", {}), "GEMINI_ENGINEER_MODEL"),
        checker_model=_model(data.get("checker_model", {}), "GEMINI_CHECKER_MODEL"),
        target_model=_model(data.get("target_model", {}), "GEMINI_TARGET_MODEL"),
        quality=QualityConfig(
            min_score=float(data.get("quality", {}).get("min_score", 0.80)),
            require_checker_pass=bool(data.get("quality", {}).get("require_checker_pass", True)),
            max_refinement_attempts=int(data.get("quality", {}).get("max_refinement_attempts", 2)),
        ),
    )
