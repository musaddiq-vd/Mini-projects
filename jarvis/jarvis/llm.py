"""Shared Gemini text-model access with automatic fallback to the next model."""

from __future__ import annotations

import threading

from google import genai
from google.genai import errors, types

from jarvis.config import TEXT_MODEL_FALLBACKS, Settings

REQUEST_TIMEOUT_MS = 40_000
# 404 = model retired, 429 = quota, 5xx = overloaded / timed out
RETRY_WITH_NEXT_MODEL = {404, 429, 500, 503, 504}


class TextModel:
    def __init__(self, settings: Settings, log=print):
        self.client = genai.Client(
            api_key=settings.effective_api_key,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
        self.models = list(dict.fromkeys([settings.text_model, *TEXT_MODEL_FALLBACKS]))
        self.log = log
        self._lock = threading.Lock()

    def generate(self, contents, config: types.GenerateContentConfig | None = None):
        """Call the first model that answers; a failing model is skipped from then on."""
        last_error: Exception | None = None
        while True:
            with self._lock:
                model = self.models[0]
                more = len(self.models) > 1
            try:
                return self.client.models.generate_content(model=model, contents=contents, config=config)
            except errors.APIError as exc:
                if exc.code not in RETRY_WITH_NEXT_MODEL or not more:
                    raise
                last_error = exc
            except Exception as exc:  # network timeout
                if not more:
                    raise
                last_error = exc
            with self._lock:
                if self.models and self.models[0] == model and len(self.models) > 1:
                    self.models.pop(0)
                    self.log(f"[model] {model} failed ({str(last_error)[:60]}), using {self.models[0]}")

    def ask(self, prompt: str, system: str | None = None) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = self.generate(prompt, config)
        return (response.text or "").strip()
