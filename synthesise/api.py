from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import load_config
from .gemini import GeminiError
from .pipeline import SynthesisPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_INDEX = BASE_DIR / "static" / "index.html"

app = FastAPI(
    title="SynThesisAI API",
    description="AI synthesis pipeline powered entirely by the Gemini API.",
    version="1.1.0",
)


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    domain: str = Field(default="general", min_length=1, max_length=100)
    count: int = Field(default=1, ge=1, le=20)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """Serve the web UI from the same FastAPI process."""
    if not STATIC_INDEX.exists():
        raise HTTPException(status_code=404, detail="Web UI not found.")
    return FileResponse(STATIC_INDEX)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "provider": "gemini"}


@app.post("/generate")
def generate(request: GenerateRequest) -> dict:
    try:
        config_path = os.getenv("SYNTHESISE_CONFIG", "config/settings.yaml")
        config = load_config(config_path)
        pipeline = SynthesisPipeline(config)
        results = pipeline.run_batch(request.topic, request.domain, request.count)
        return {
            "count": len(results),
            "results": [asdict(result) for result in results],
        }
    except GeminiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
