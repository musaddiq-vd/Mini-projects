"""Little sci-fi sound effects, synthesised with numpy (no audio files needed).

Sounds: startup, wake, sleep, action, success, error, notify, alert.
`SoundFX.attach(events)` plays them on their own when things happen.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

RATE = 24_000


def _env(n: int, attack: float = 0.01, release: float = 0.08) -> np.ndarray:
    """Attack / exponential-decay envelope for n samples."""
    t = np.arange(n) / RATE
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    r = np.exp(-t / max(release, 1e-4))
    return a * r


def _tone(freq: float, dur: float, release: float = 0.12, harmonics=(1.0, 0.35, 0.12),
          sweep_to: float | None = None) -> np.ndarray:
    n = int(RATE * dur)
    t = np.arange(n) / RATE
    if sweep_to:
        f = np.linspace(freq, sweep_to, n)
        phase = 2 * np.pi * np.cumsum(f) / RATE
    else:
        phase = 2 * np.pi * freq * t
    wave = sum(amp * np.sin(phase * (k + 1)) for k, amp in enumerate(harmonics))
    return wave * _env(n, release=release)


def _seq(*parts: np.ndarray, gap: float = 0.0) -> np.ndarray:
    silence = np.zeros(int(RATE * gap))
    out = []
    for p in parts:
        out += [p, silence]
    return np.concatenate(out[:-1]) if out else np.zeros(1)


def _mix(*parts: np.ndarray) -> np.ndarray:
    n = max(len(p) for p in parts)
    out = np.zeros(n)
    for p in parts:
        out[: len(p)] += p
    return out


def _sparkle(dur: float, base: float = 2400) -> np.ndarray:
    rng = np.random.default_rng(7)
    n = int(RATE * dur)
    out = np.zeros(n)
    for _ in range(10):
        start = rng.integers(0, max(1, n - RATE // 20))
        ping = _tone(base * rng.uniform(0.8, 1.6), 0.05, release=0.02, harmonics=(1.0,)) * 0.25
        out[start: start + len(ping)] += ping[: n - start]
    return out


def _noise(dur: float, amp: float = 0.5) -> np.ndarray:
    n = int(RATE * dur)
    return np.random.default_rng(3).uniform(-amp, amp, n) * _env(n, attack=0.002, release=dur / 3)


def _pad(sound: np.ndarray, before: float) -> np.ndarray:
    return np.concatenate([np.zeros(int(RATE * before)), sound])


def build_sounds() -> dict[str, np.ndarray]:
    s = {
        "startup": _mix(_tone(260, 0.55, release=0.3, sweep_to=1040), _sparkle(0.55) * 0.6),
        "wake": _seq(_tone(660, 0.09, release=0.06), _tone(990, 0.16, release=0.1), gap=0.015),
        "sleep": _seq(_tone(880, 0.1, release=0.07), _tone(587, 0.12, release=0.08),
                      _tone(392, 0.25, release=0.15), gap=0.01),
        "action": _tone(1320, 0.06, release=0.025, harmonics=(1.0, 0.2)),
        "success": _seq(_tone(1046, 0.05, release=0.03), _tone(1568, 0.09, release=0.05), gap=0.005),
        "error": _seq(_tone(196, 0.12, release=0.08, harmonics=(1.0, 0.6, 0.4, 0.3)),
                      _tone(165, 0.16, release=0.1, harmonics=(1.0, 0.6, 0.4, 0.3)), gap=0.03),
        "notify": _seq(_tone(784, 0.14, release=0.12), _tone(988, 0.14, release=0.12),
                       _tone(1175, 0.35, release=0.25), gap=0.02),
        "alarm": _seq(*[_seq(_tone(1318, 0.12, release=0.08), _tone(1046, 0.12, release=0.08), gap=0.02)
                        for _ in range(3)], gap=0.12),
        "love": _seq(_tone(1568, 0.07, release=0.05), _tone(2093, 0.12, release=0.08), gap=0.01),
        "shutter": _mix(_noise(0.05, 0.6), _tone(2400, 0.04, release=0.015, harmonics=(1.0,)),
                        _pad(_noise(0.07, 0.4), 0.09)),
        "alert": _seq(_tone(880, 0.09, release=0.05), _tone(880, 0.09, release=0.05),
                      _tone(1175, 0.18, release=0.1), gap=0.06),
    }
    for k, v in s.items():
        peak = float(np.max(np.abs(v))) or 1.0
        s[k] = (v / peak * 0.6).astype(np.float32)
    return s


class SoundFX:
    """Plays effects one after another on a background thread. Never raises."""

    def __init__(self, enabled: bool = True, volume: int = 60, device: int | None = None):
        self.enabled = enabled
        self.volume = volume
        self.device = device
        self.sounds = build_sounds()
        self.busy_until = 0.0  # monotonic time until an effect is audible (echo guard)
        self._q: queue.Queue[str | None] = queue.Queue(maxsize=8)
        self._thread: threading.Thread | None = None
        self._prev_state = ""
        self._started = False

    def duration(self, name: str) -> float:
        return len(self.sounds.get(name, ())) / RATE

    def play(self, name: str) -> None:
        if not self.enabled or name not in self.sounds or self.volume <= 0:
            return
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="jarvis-sfx", daemon=True)
            self._thread.start()
        self.busy_until = max(self.busy_until, time.monotonic()) + self.duration(name) + 0.15
        try:
            self._q.put_nowait(name)
        except queue.Full:
            pass

    def _run(self) -> None:
        import sounddevice as sd

        while True:
            name = self._q.get()
            if name is None:
                return
            data = self.sounds[name] * (min(self.volume, 100) / 100.0)
            try:
                with sd.OutputStream(samplerate=RATE, channels=1, dtype="float32", device=self.device) as out:
                    out.write(data.reshape(-1, 1))
            except Exception:
                pass  # no speakers / device busy: effects are optional

    def close(self) -> None:
        if self._thread:
            try:
                self._q.put_nowait(None)
            except queue.Full:
                pass

    # -- automatic effects ------------------------------------------------------
    def on_event(self, kind: str, data: dict) -> None:
        if kind == "state":
            state, prev = data["state"], self._prev_state
            self._prev_state = state
            awake = state in {"listening", "speaking", "muted", "thinking"}
            if not self._started:
                if state not in {"connecting", "offline"}:
                    self._started = True
                    self.play("startup")
            elif state == "sleeping" and prev != "sleeping":
                self.play("sleep")
            elif prev == "sleeping" and awake:
                self.play("wake")
        elif kind == "tool_start":
            self.play("action")
        elif kind == "tool":
            self.play("error" if "error" in data.get("result", {}) else "success")
        elif kind == "log" and str(data.get("text", "")).startswith("🔔"):
            self.play("notify")
        elif kind == "capture":
            self.play("shutter")
        elif kind in ("confirm_needed", "guardrail"):
            self.play("alert")
        elif kind == "timer_end" and data.get("done"):
            self.play("alarm")
        elif kind == "emote" and data.get("feeling") == "love":
            self.play("love")

    def attach(self, events) -> "SoundFX":
        events.subscribe(self.on_event)
        return self
