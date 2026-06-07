from __future__ import annotations

import json
import os
from typing import Protocol
from urllib import error, request


class ModelBackend(Protocol):
    """Optional model backend for natural-language reasoning."""

    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class DeterministicBackend:
    """A no-network backend used for reproducible local runs."""

    def complete(self, prompt: str) -> str:
        return prompt


class BackendError(RuntimeError):
    """Raised when a model backend cannot return usable text."""


class OpenAIResponsesBackend:
    """OpenAI Responses API backend using only the Python standard library."""

    endpoint = "https://api.openai.com/v1/responses"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
        self.timeout_seconds = timeout_seconds
        if not self.api_key:
            raise BackendError("OPENAI_API_KEY is required for OpenAIResponsesBackend.")

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "instructions": (
                "You improve coding-agent task prompts. Return only the improved prompt text. "
                "Use these exact section headers once: Objective, Context, Constraints, Verification, Done."
            ),
            "input": prompt,
            "max_output_tokens": 800,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (OSError, error.HTTPError, json.JSONDecodeError) as exc:
            raise BackendError(f"OpenAI Responses API request failed: {exc}") from exc

        text = self._extract_text(body)
        if not text.strip():
            raise BackendError("OpenAI Responses API returned no text.")
        return text.strip()

    def _extract_text(self, body: dict[str, object]) -> str:
        output_text = body.get("output_text")
        if isinstance(output_text, str):
            return output_text

        chunks: list[str] = []
        output = body.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
        return "\n".join(chunks)
