"""Settings, stored as JSON in the Jarvis home folder (never inside the repo).

The home folder is ``$JARVIS_HOME`` if set, otherwise ``~/.jarvis``.
The API key is read from ``$GEMINI_API_KEY`` first, then from the settings file.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

VOICES = ("Charon", "Puck", "Kore", "Fenrir", "Aoede")

# Text models tried in order when one is retired, overloaded or slow.
TEXT_MODEL_FALLBACKS = ("gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite")
# Retired models and old Jarvis defaults; config files naming them get the current default.
OUTDATED_MODELS = {"gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash",
                   "gemini-3.1-flash-live-preview"}


def home_dir() -> Path:
    custom = os.environ.get("JARVIS_HOME")
    path = Path(custom) if custom else Path.home() / ".jarvis"
    if not custom and not path.exists():
        _move_old_home(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _move_old_home(new: Path) -> None:
    """The project used to be called Orion: carry ~/.orion (key, memory) over once."""
    old = Path.home() / ".orion"
    if old.is_dir():
        import shutil

        shutil.copytree(old, new)


@dataclass
class Settings:
    api_key: str = ""
    assistant_name: str = "Jarvis"
    user_name: str = ""
    voice: str = "Charon"
    live_model: str = "gemini-3.8-live"
    text_model: str = "gemini-3.5-flash"
    input_device: str = ""  # device name; empty = system default
    output_device: str = ""
    wake_word: str = ""  # openwakeword model name, e.g. "hey_jarvis"; empty = off
    sleep_after_seconds: int = 120
    # Folders the file tools may touch. Empty = your home folder.
    allowed_roots: list[str] = field(default_factory=list)
    # Look and behaviour
    theme_color: str = "#00d4ff"
    allow_interrupt: bool = False  # talk over Jarvis (needs headphones)
    morning_briefing: bool = True  # greet + weather + news on the first start of the day
    proactive: bool = True  # friendly check-in after a long quiet spell
    proactive_minutes: int = 60
    hardware_alerts: bool = True  # warn about high CPU/RAM, low battery
    city: str = ""  # for weather in the briefing; falls back to memory

    @property
    def effective_api_key(self) -> str:
        return os.environ.get("GEMINI_API_KEY", "").strip() or self.api_key.strip()

    @property
    def roots(self) -> list[Path]:
        if not self.allowed_roots:
            return [Path.home().resolve()]
        return [Path(p).expanduser().resolve() for p in self.allowed_roots]


def settings_path() -> Path:
    return home_dir() / "config.json"


def load_settings(path: Path | None = None) -> Settings:
    path = path or settings_path()
    if not path.exists():
        return Settings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return Settings()
    known = {f.name for f in fields(Settings)}
    data = {k: v for k, v in data.items() if k in known}
    for key in ("text_model", "live_model"):
        if data.get(key) in OUTDATED_MODELS:
            data.pop(key)
    return Settings(**data)


def save_settings(settings: Settings, path: Path | None = None) -> Path:
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # Only store what differs from the defaults, so improved defaults reach old installs.
    defaults = asdict(Settings())
    data = {k: v for k, v in asdict(settings).items() if v != defaults[k]}
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path
