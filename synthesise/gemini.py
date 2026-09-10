from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class GeminiError(RuntimeError):
    """Raised when Gemini cannot produce a usable response."""


@dataclass
class GeminiClient:
    model: str
    temperature: float = 0.4
    max_output_tokens: int = 2048
    retries: int = 3
    backoff_seconds: float = 1.5

    def __post_init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise GeminiError("GEMINI_API_KEY is not set. Put a new key in your local .env file.")
        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, *, system_instruction: str | None = None) -> str:
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            system_instruction=system_instruction,
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                text = getattr(response, "text", None)
                if text:
                    return text.strip()
                raise GeminiError("Gemini returned no text content.")
            except Exception as exc:  # SDK exceptions vary by version.
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
        raise GeminiError(f"Gemini request failed after {self.retries} attempts: {last_error}")

    def generate_json(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
    ) -> dict[str, Any]:
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            system_instruction=system_instruction,
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                text = getattr(response, "text", None)
                if not text:
                    raise GeminiError("Gemini returned an empty JSON response.")
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise GeminiError("Expected a JSON object from Gemini.")
                return value
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
        raise GeminiError(f"Gemini JSON request failed: {last_error}")
