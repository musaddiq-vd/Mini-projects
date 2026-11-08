"""Settings, stored as JSON in the Jarvis home folder (never inside the repo).

The home folder is ``$JARVIS_HOME`` if set, otherwise ``~/.jarvis``.
The API key is read from ``$GEMINI_API_KEY`` first, then from the settings file.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# Gemini voices and how they sound (the first five were the original choices).
VOICE_INFO = {
    "Charon": "informative, deep", "Puck": "upbeat", "Kore": "firm", "Fenrir": "excitable", "Aoede": "breezy",
    "Zephyr": "bright", "Leda": "youthful", "Orus": "firm, male", "Callirrhoe": "easy-going",
    "Autonoe": "bright", "Enceladus": "breathy", "Iapetus": "clear", "Umbriel": "easy-going",
    "Algieba": "smooth", "Despina": "smooth", "Erinome": "clear", "Algenib": "gravelly",
    "Rasalgethi": "informative", "Laomedeia": "upbeat", "Achernar": "soft", "Alnilam": "firm",
    "Schedar": "even", "Gacrux": "mature", "Pulcherrima": "forward", "Achird": "friendly",
    "Zubenelgenubi": "casual", "Vindemiatrix": "gentle", "Sadachbia": "lively", "Sadaltager": "knowledgeable",
    "Sulafat": "warm",
}
VOICES = tuple(VOICE_INFO)

# How Jarvis talks. Each style is a short instruction added to the system prompt.
SPEAKING_STYLES = {
    "friendly": "warm, relaxed and caring, like a close friend who is genuinely happy to help",
    "calm": "calm, soft and unhurried, with a soothing, reassuring tone",
    "energetic": "upbeat, lively and enthusiastic, with real energy in your voice",
    "professional": "polite, crisp and confident, like a top-class personal assistant",
    "funny": "playful and witty, with light jokes and teasing when the moment allows (never at the user's expense)",
}

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
    theme_color: str = "#00d4ff"  # custom accent; a theme's own accent when it matches a theme
    theme_name: str = "Jarvis Blue"
    theme_mode: str = "dark"  # dark | light | auto (light by day, dark at night)
    allow_interrupt: bool = False  # talk over Jarvis (needs headphones)
    morning_briefing: bool = True  # greet + weather + news on the first start of the day
    proactive: bool = True  # friendly check-in after a long quiet spell
    proactive_minutes: int = 60
    hardware_alerts: bool = True  # warn about high CPU/RAM, low battery
    city: str = ""  # for weather in the briefing; falls back to memory
    # Say the assistant's name to wake it (works with any name and language).
    # Uses the online session while asleep; the offline wake word above is used instead when set.
    name_wake: bool = True
    sound_effects: bool = True
    sound_volume: int = 60  # 0..100
    orb_face: bool = True  # AI face on the orb core
    # wellbeing
    evening_recap: bool = True  # a short recap of the day in the evening
    recap_hour: int = 21
    health_nudges: bool = True  # eye breaks and water reminders while you use the PC
    eye_break_minutes: int = 50
    water_minutes: int = 90
    # privacy
    save_history: bool = True  # keep the conversation on this computer (Settings > Privacy)
    history_days: int = 30  # delete saved conversation older than this (0 = keep forever)
    ask_camera: bool = True  # ask before Jarvis takes a photo with the webcam
    notify_capture: bool = True  # flash + notice whenever Jarvis looks at the screen or camera
    speaking_style: str = "friendly"  # see SPEAKING_STYLES
    expressive_voice: bool = True  # the voice follows the mood of the conversation (affective dialog)

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
    from jarvis import secure

    stored_key = data.get("api_key", "")
    data["api_key"] = secure.unprotect(stored_key)
    settings = Settings(**data)
    if stored_key and not stored_key.startswith(secure.PREFIX) and secure.available():
        save_settings(settings, path)  # an old plain-text key: encrypt it on disk right away
    return settings


def save_settings(settings: Settings, path: Path | None = None) -> Path:
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # Only store what differs from the defaults, so improved defaults reach old installs.
    from jarvis import secure

    defaults = asdict(Settings())
    data = {k: v for k, v in asdict(settings).items() if v != defaults[k]}
    if data.get("api_key"):
        data["api_key"] = secure.protect(data["api_key"])  # never stored in plain text on Windows
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path
