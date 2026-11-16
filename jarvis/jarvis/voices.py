"""Voice profiles: tell who is speaking from their voice, and use that person's settings.

How it works (no extra packages, CPU only, a few milliseconds per turn):
- Enrollment records ~3 short phrases. From each one we take MFCCs (the usual "shape of the voice"
  features), and keep their per-coefficient mean and spread. Only these numbers are saved, in
  ~/.jarvis/voices/<name>.json - the raw audio is thrown away right after.
- For every user turn, the voiced part of the microphone audio is turned into the same numbers on a
  background thread and compared with each profile (a distance scaled by how much that person's own
  phrases varied). A clear winner switches the active profile; a voice far from everyone is a guest.

Honest accuracy note: this is classic MFCC statistics, not a neural speaker model (those need PyTorch,
hundreds of MB). In a quiet room it tells apart a few people with clearly different voices (e.g. an adult
and a child, or a man and a woman) quite reliably; similar voices, a different microphone, background noise
or a cold can confuse it. It is a convenience to pick the right name/language/style - never a security
check, which is why guests are only kept away from WhatsApp, contacts and money, and the owner can allow even
that in Settings.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np

log = logging.getLogger("jarvis.voices")

RATE = 16_000
N_MFCC = 20  # coefficient 0 (loudness) is dropped
N_MELS = 26
FRAME = 400  # 25 ms
HOP = 160  # 10 ms
NFFT = 512
MIN_VOICED_SECONDS = 1.0  # less speech than this isn't enough to judge
MAX_TURN_SECONDS = 8.0  # only the last few seconds of a turn are used
PHRASE_SECONDS = 4.0
PHRASES = ("Hey Jarvis, what's the weather like today?",
           "Please remind me to call my mother at six in the evening.",
           "The quick brown fox jumps over the lazy dog.")
MATCH = 2.8  # distance below this: it's this person
GUEST = 3.5  # every profile further than this: an unknown voice
MARGIN = 1.2  # the second best must be at least this many times further away
STYLE_NAMES = ("friendly", "calm", "energetic", "professional", "funny")
MEMORY_CHOICES = ("all", "own", "none")
GUEST_TOOLS = {  # phone, WhatsApp and money: not for unknown voices unless the owner allows it
    "send_whatsapp", "find_contact", "save_contact", "delete_contact",
    "add_expense", "expense_summary", "list_expenses", "delete_expense", "set_budget",
}


def voices_dir() -> Path:
    from jarvis.config import home_dir

    path = home_dir() / "voices"
    path.mkdir(parents=True, exist_ok=True)
    return path


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")[:40] or "voice"


# -- features --------------------------------------------------------------------------------------
def _mel_bank() -> np.ndarray:
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mels = np.linspace(hz_to_mel(60), hz_to_mel(RATE / 2 - 200), N_MELS + 2)
    bins = np.floor((NFFT + 1) * mel_to_hz(mels) / RATE).astype(int)
    bank = np.zeros((N_MELS, NFFT // 2 + 1), dtype=np.float32)
    for i in range(1, N_MELS + 1):
        left, mid, right = bins[i - 1], bins[i], bins[i + 1]
        for k in range(left, mid):
            bank[i - 1, k] = (k - left) / max(1, mid - left)
        for k in range(mid, right):
            bank[i - 1, k] = (right - k) / max(1, right - mid)
    return bank


_BANK = _mel_bank()
_DCT = np.cos(np.pi / N_MELS * (np.arange(N_MELS)[None, :] + 0.5) * np.arange(N_MFCC)[:, None]).astype(np.float32)
_WINDOW = np.hamming(FRAME).astype(np.float32)


def as_float(audio) -> np.ndarray:
    """int16 bytes / int16 array / float array -> float32 mono in -1..1."""
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(audio), dtype=np.int16).astype(np.float32) / 32768.0
    a = np.asarray(audio)
    if a.dtype.kind in "iu":
        return a.astype(np.float32) / 32768.0
    a = a.astype(np.float32, copy=False)
    return a.mean(axis=1) if a.ndim == 2 else a


def _frames(x: np.ndarray) -> np.ndarray:
    if len(x) < FRAME:
        return np.zeros((0, FRAME), dtype=np.float32)
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def voiced_frames(x: np.ndarray) -> np.ndarray:
    """The frames with speech in them (simple energy gate relative to the loudest part)."""
    frames = _frames(x)
    if not len(frames):
        return frames
    energy = np.sqrt((frames ** 2).mean(axis=1))
    loud = np.percentile(energy, 95)
    keep = (energy > max(0.008, loud * 0.15))
    return frames[keep]


def mfcc(frames: np.ndarray) -> np.ndarray:
    """(n, FRAME) frames -> (n, N_MFCC - 1) MFCCs without c0."""
    if not len(frames):
        return np.zeros((0, N_MFCC - 1), dtype=np.float32)
    emph = np.concatenate([frames[:, :1], frames[:, 1:] - 0.97 * frames[:, :-1]], axis=1)
    spec = np.abs(np.fft.rfft(emph * _WINDOW, NFFT)) ** 2 / NFFT
    logmel = np.log(spec @ _BANK.T + 1e-8)
    return (logmel @ _DCT.T)[:, 1:]


def voiced_seconds(audio) -> float:
    return len(voiced_frames(as_float(audio))) * HOP / RATE


def features(audio) -> np.ndarray | None:
    """One voice "fingerprint" for a piece of speech: MFCC means and spreads. None if too little speech."""
    frames = voiced_frames(as_float(audio))
    if len(frames) * HOP / RATE < MIN_VOICED_SECONDS:
        return None
    c = mfcc(frames)
    return np.concatenate([c.mean(axis=0), c.std(axis=0)]).astype(np.float64)


def build_profile(samples: list) -> dict:
    """The saved numbers for a person from their enrollment phrases."""
    feats = [f for f in (features(s) for s in samples) if f is not None]
    if not feats:
        raise ValueError("not enough speech")
    stack = np.stack(feats)
    mean = stack.mean(axis=0)
    spread = stack.std(axis=0) if len(feats) > 1 else np.zeros_like(mean)
    # a floor: three phrases can't show all of a voice's natural variation
    scale = np.maximum(spread, 0.1 * np.abs(mean) + 0.3)
    return {"mean": [round(float(v), 5) for v in mean], "scale": [round(float(v), 5) for v in scale],
            "phrases": len(feats)}


def distance(feat: np.ndarray, profile: dict) -> float:
    mean = np.asarray(profile["mean"], dtype=np.float64)
    scale = np.asarray(profile["scale"], dtype=np.float64)
    if mean.shape != feat.shape:
        return float("inf")
    return float(np.sqrt(np.mean(((feat - mean) / scale) ** 2)))


# -- profiles on disk ------------------------------------------------------------------------------
def load_profiles() -> list[dict]:
    out = []
    for f in sorted(voices_dir().glob("*.json")):
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(p, dict) and p.get("name") and p.get("voice"):
            p["id"] = f.stem
            out.append(p)
    return out


def get_profile(name: str) -> dict | None:
    key = slug(name)
    for p in load_profiles():
        if p["id"] == key or p["name"].lower() == (name or "").strip().lower():
            return p
    return None


def save_profile(profile: dict) -> Path:
    path = voices_dir() / f"{slug(profile['name'])}.json"
    data = {k: v for k, v in profile.items() if k != "id"}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def delete_profile(name: str) -> bool:
    p = get_profile(name)
    if p is None:
        return False
    (voices_dir() / f"{p['id']}.json").unlink(missing_ok=True)
    if _active.get("id") == p["id"]:
        set_active(None, guest=False)
    return True


def owner_profile(profiles: list[dict] | None = None) -> dict | None:
    profiles = load_profiles() if profiles is None else profiles
    return next((p for p in profiles if p.get("owner")), profiles[0] if profiles else None)


def enroll(name: str, samples: list, language: str = "auto", style: str = "", memory: str = "",
           owner: bool | None = None) -> dict:
    """Make (or replace) a profile from recorded phrases. The audio itself is not kept."""
    voice = build_profile(samples)
    existing = get_profile(name)
    profiles = load_profiles()
    if owner is None:
        owner = bool(existing.get("owner")) if existing else not any(p.get("owner") for p in profiles)
    if owner:  # one owner only
        for p in profiles:
            if p.get("owner") and p["id"] != slug(name):
                p["owner"] = False
                save_profile(p)
    profile = {
        "name": name.strip(), "owner": bool(owner), "language": language or "auto",
        "style": style if style in STYLE_NAMES else "",
        "memory": memory if memory in MEMORY_CHOICES else ("all" if owner else "own"),
        "voice": voice, "created": (existing or {}).get("created", time.time()), "updated": time.time(),
    }
    save_profile(profile)
    profile["id"] = slug(name)
    return profile


# -- who is speaking -------------------------------------------------------------------------------
_lock = threading.Lock()
_active: dict = {}  # {"id", "name", "guest", "score", "at"}; empty = nobody identified yet (owner defaults)


def set_active(profile: dict | None, guest: bool = False, score: float = 0.0) -> None:
    with _lock:
        _active.clear()
        if profile is not None:
            _active.update(id=profile["id"], name=profile["name"], guest=False, score=score, at=time.time())
        elif guest:
            _active.update(id="", name="guest", guest=True, score=score, at=time.time())


def active() -> dict:
    with _lock:
        return dict(_active)


def is_guest() -> bool:
    with _lock:
        return bool(_active.get("guest"))


def identify(audio, profiles: list[dict] | None = None) -> dict:
    """{"name", "id", "distance", "guest", "sure"} for a piece of speech; name "" when it can't tell."""
    profiles = load_profiles() if profiles is None else profiles
    feat = features(audio)
    if feat is None or not profiles:
        return {"name": "", "id": "", "distance": None, "guest": False, "sure": False}
    scored = sorted(((distance(feat, p["voice"]), p) for p in profiles), key=lambda t: t[0])
    best_d, best = scored[0]
    second = scored[1][0] if len(scored) > 1 else float("inf")
    if best_d > GUEST:
        return {"name": "guest", "id": "", "distance": round(best_d, 2), "guest": True, "sure": True}
    sure = best_d <= MATCH and second >= best_d * MARGIN
    return {"name": best["name"], "id": best["id"], "distance": round(best_d, 2), "guest": False, "sure": sure}


def system_note(profile: dict | None, guest: bool, settings) -> str:
    """What the model is told when the speaker changes."""
    owner = owner_profile()
    owner_name = (owner or {}).get("name") or getattr(settings, "user_name", "") or "the owner"
    if guest:
        return ("(system: voice) The voice speaking now doesn't match any saved voice profile - treat them as a "
                f"guest. Don't call them {owner_name}, and don't share {owner_name}'s personal facts, contacts, "
                "messages or money details. WhatsApp, contacts and money tools are off for guests unless the owner "
                "allowed it. Be friendly; if they tell you their name, use it.")
    from jarvis.config import SPEAKING_STYLES
    from jarvis.lang import label

    name = profile["name"]
    parts = [f"(system: voice) {name} is speaking now (recognised by voice). Address them as {name}."]
    if profile.get("language") and profile["language"] != "auto":
        parts.append(f"Reply in {label(profile['language'])}.")
    if profile.get("style") in SPEAKING_STYLES:
        parts.append(f"Your manner with them: {SPEAKING_STYLES[profile['style']]}.")
    memory = profile.get("memory", "all")
    if profile.get("owner"):
        parts.append("They are the owner: use their saved facts as usual.")
    elif memory == "none":
        parts.append(f"Don't use or share any saved personal facts (they belong to {owner_name}).")
    elif memory == "own":
        parts.append(f"Only use saved facts about {name} (keys mentioning '{slug(name)}'); keep {owner_name}'s "
                     "private facts to yourself.")
    return " ".join(parts)


def prompt_line() -> str:
    """For the system prompt after a reconnect: who was speaking last."""
    a = active()
    if not a:
        return ""
    if a.get("guest"):
        return "Voice profile: the last voice heard was a guest (not a saved profile)."
    return f"Voice profile: the last voice heard was {a['name']}."


class VoiceWatcher:
    """Collects the mic audio of each user turn and identifies the speaker on a background thread.

    `feed` is called for every mic chunk that is really sent (not muted, not echo), `end_turn` when a user
    turn is finished. Nothing here ever blocks the conversation: if the last check is still running, a turn
    is skipped.
    """

    def __init__(self, settings, events=None, on_change: Callable[[dict | None, bool, dict], None] | None = None,
                 identify_fn: Callable = identify, background: bool = True):
        self.s = settings
        self.events = events
        self.on_change = on_change
        self.identify_fn = identify_fn
        self.background = background
        self._buf: deque = deque()
        self._len = 0
        self._busy = threading.Lock()
        self._profiles: list[dict] | None = None
        self._profiles_at = 0.0
        self._pending: dict | None = None  # a change seen once, waiting for a second turn to agree

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.s, "voice_id", True))

    def profiles(self) -> list[dict]:
        if self._profiles is None or time.monotonic() - self._profiles_at > 30:
            self._profiles = load_profiles()
            self._profiles_at = time.monotonic()
        return self._profiles

    def reload(self) -> None:
        self._profiles = None

    def feed(self, chunk) -> None:
        if not self.enabled:
            return
        self._buf.append(chunk)
        self._len += len(chunk) // 2 if isinstance(chunk, (bytes, bytearray)) else len(chunk)
        while self._len > MAX_TURN_SECONDS * RATE and len(self._buf) > 1:
            old = self._buf.popleft()
            self._len -= len(old) // 2 if isinstance(old, (bytes, bytearray)) else len(old)

    def drop(self) -> None:
        """Forget the audio collected so far (the turn had no words, or Jarvis was asleep)."""
        self._buf.clear()
        self._len = 0

    def end_turn(self) -> None:
        chunks = list(self._buf)
        self._buf.clear()
        self._len = 0
        if not self.enabled or not chunks or not self.profiles():
            return
        if not self._busy.acquire(blocking=False):
            return  # the last check is still running: skip this turn
        audio = np.concatenate([as_float(c) for c in chunks])[-int(MAX_TURN_SECONDS * RATE):]
        if self.background:
            threading.Thread(target=self._check, args=(audio,), daemon=True, name="jarvis-voice-id").start()
        else:
            self._check(audio)

    def _check(self, audio: np.ndarray) -> None:
        try:
            result = self.identify_fn(audio, self.profiles())
            if not result.get("name"):
                return
            if self.events is not None:
                self.events.emit("speaker", **result)
            self._decide(result)
        except Exception:
            log.exception("voice check failed")
        finally:
            self._busy.release()

    def _decide(self, result: dict) -> None:
        now = active()
        same = (result["guest"] and now.get("guest")) or (not result["guest"] and now.get("id") == result["id"])
        if same:
            self._pending = None
            return
        if not result["sure"]:
            return
        if not now:
            needed = 1  # nobody known yet: the first sure answer counts
        else:
            needed = 2  # a change needs two turns in a row to agree (one odd turn doesn't flip it)
        key = "guest" if result["guest"] else result["id"]
        if needed == 2 and (self._pending or {}).get("key") != key:
            self._pending = {"key": key}
            return
        self._pending = None
        profile = None if result["guest"] else next((p for p in self.profiles() if p["id"] == result["id"]), None)
        if profile is None and not result["guest"]:
            return
        previous = active()
        set_active(profile, guest=result["guest"], score=result.get("distance") or 0.0)
        if self.events is not None:
            self.events.emit("voice_profile", name=result["name"], guest=result["guest"])
        if self.on_change is not None:
            self.on_change(profile, result["guest"], previous)
