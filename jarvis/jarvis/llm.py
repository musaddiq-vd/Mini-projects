"""Shared Gemini text-model access with automatic fallback to the next model."""

from __future__ import annotations

import json
import threading
import time

from google import genai
from google.genai import errors, types

from jarvis.config import TEXT_MODEL_FALLBACKS, Settings

REQUEST_TIMEOUT_MS = 25_000  # a normal answer takes 1-8 s; don't hang on an overloaded model
# 404 = model retired, 429 = quota, 5xx = overloaded / timed out
RETRY_WITH_NEXT_MODEL = {404, 429, 500, 503, 504}
# how long a model that failed is skipped, also after a restart (so every start doesn't wait for it again)
COOL_DOWN_S = {404: 7 * 86400, 429: 30 * 60}
COOL_DOWN_OTHER_S = 10 * 60  # overloaded / timed out


class QuotaReached(RuntimeError):
    """Every model hit Gemini's usage limit. The message is meant for the user."""

    def __init__(self):
        super().__init__("Gemini's free usage limit is used up for now. Try again in a few minutes "
                         "(or add billing to the API key for higher limits).")


def _health_path():
    from jarvis.config import home_dir

    return home_dir() / "model_health.json"


def _load_health() -> dict[str, float]:
    try:
        data = json.loads(_health_path().read_text(encoding="utf-8"))
        return {str(k): float(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def mark_failed(model: str, code: int | None) -> None:
    """Skip this model for a while (30 min after a quota error, a week after 'not found', 10 min otherwise)."""
    data = {k: v for k, v in _load_health().items() if v > time.time()}
    data[model] = time.time() + COOL_DOWN_S.get(code or 0, COOL_DOWN_OTHER_S)
    try:
        path = _health_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass


def usable_order(models: list[str], now: float | None = None) -> list[str]:
    """Models that are fine first, cooling-down ones last (still tried if all else fails)."""
    now = time.time() if now is None else now
    cooling = {m for m, until in _load_health().items() if until > now}
    return [m for m in models if m not in cooling] + [m for m in models if m in cooling]


class TextModel:
    def __init__(self, settings: Settings, log=print):
        self.client = genai.Client(
            api_key=settings.effective_api_key,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
        self.models = usable_order(list(dict.fromkeys([settings.text_model, *TEXT_MODEL_FALLBACKS])))
        self.log = log
        self._lock = threading.Lock()

    def generate(self, contents, config: types.GenerateContentConfig | None = None):
        """Call the first model that answers; a failing model is skipped from then on."""
        last_error: Exception | None = None
        retried_last = False
        while True:
            with self._lock:
                model = self.models[0]
                more = len(self.models) > 1
            try:
                return self.client.models.generate_content(model=model, contents=contents, config=config)
            except errors.APIError as exc:
                if exc.code in (500, 503, 504) and not more and not retried_last:
                    retried_last = True  # the last model was busy for a moment: one more try
                    time.sleep(1.5)
                    continue
                if exc.code == 429 and not more:
                    mark_failed(model, 429)
                    raise QuotaReached() from exc
                if exc.code not in RETRY_WITH_NEXT_MODEL or not more:
                    raise
                last_error = exc
                mark_failed(model, exc.code)
            except Exception as exc:  # network timeout
                if not more:
                    raise
                last_error = exc
                mark_failed(model, None)
            with self._lock:
                if self.models and self.models[0] == model and len(self.models) > 1:
                    self.models.pop(0)
                    self.log(f"[model] {model} failed ({str(last_error)[:60]}), using {self.models[0]}")

    def ask_image(self, prompt: str, jpeg: bytes, system: str | None = None) -> str:
        """Ask about a picture (a screenshot). Returns the text answer."""
        config = types.GenerateContentConfig(
            system_instruction=system,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        parts = [types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt]
        response = self.generate(parts, config)
        return (response.text or "").strip()

    def transcribe(self, audio: bytes, mime: str = "audio/ogg", prompt: str = "") -> str:
        """Speech in a voice note -> text (in the language it was spoken). `prompt` replaces the instruction."""
        config = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        parts = [types.Part.from_bytes(data=audio, mime_type=mime),
                 prompt or "Write down exactly what is said in this voice message, in the language it is spoken. "
                           "Only the words - no notes. If nothing is said, reply with nothing."]
        response = self.generate(parts, config)
        return (response.text or "").strip()

    def ask(self, prompt: str, system: str | None = None) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = self.generate(prompt, config)
        return (response.text or "").strip()
