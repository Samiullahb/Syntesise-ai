from __future__ import annotations

import os
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_config
from .gemini import GeminiError
from .pipeline import SynthesisPipeline

app = FastAPI(
    title="SynThesisAI API",
    description="AI synthesis pipeline powered by Gemini 3.1 Pro Preview.",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    domain: str = Field(default="general", min_length=1, max_length=100)
    count: int = Field(default=1, ge=1, le=20)


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "SynThesisAI", "status": "online", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/generate")
def generate(request: GenerateRequest) -> dict:
    try:
        config = load_config(os.getenv("SYNTHESISE_CONFIG", "config/settings.yaml"))
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
