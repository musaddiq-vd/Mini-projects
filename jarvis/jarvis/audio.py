"""Microphone and speaker helpers (sounddevice)."""

from __future__ import annotations

import queue
import threading

import numpy as np
import sounddevice as sd

MIC_RATE = 16_000      # Gemini Live expects 16 kHz 16-bit mono input
SPEAKER_RATE = 24_000  # and sends 24 kHz 16-bit mono output
MIC_BLOCK = 1_280      # 80 ms - also the frame size openwakeword wants


def list_devices() -> list[dict]:
    default_in, default_out = sd.default.device
    rows = []
    for i, d in enumerate(sd.query_devices()):
        host = sd.query_hostapis(d["hostapi"])["name"]
        rows.append({
            "index": i, "name": d["name"], "host": host,
            "inputs": d["max_input_channels"], "outputs": d["max_output_channels"],
            "default": i in (default_in, default_out),
        })
    return rows


def find_device(name: str, kind: str) -> int | None:
    """Device index by (part of) its name; None means system default."""
    if not name:
        return None
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, d in enumerate(sd.query_devices()):
        if d[key] > 0 and name.lower() in d["name"].lower():
            return i
    print(f"[audio] '{name}' not found, using the default {kind} device.")
    return None


def level(pcm16: bytes) -> float:
    """Rough loudness 0..1 of a 16-bit PCM chunk, for the UI animation."""
    if not pcm16:
        return 0.0
    samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    rms = float(np.sqrt(np.mean(samples * samples))) / 32768.0
    return min(1.0, rms * 6)


class Speaker:
    """Plays PCM chunks on a background thread; can be flushed on interruption."""

    def __init__(self, device: int | None = None, on_level=None):
        self._on_level = on_level  # called with 0..1 loudness of each chunk played
        self._q: queue.Queue[bytes | None] = queue.Queue()
        self._stream = sd.RawOutputStream(samplerate=SPEAKER_RATE, channels=1, dtype="int16", device=device)
        self._busy = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._stream.start()
        self._thread.start()

    def _run(self) -> None:
        while True:
            chunk = self._q.get()
            if chunk is None:
                return
            self._busy.set()
            if self._on_level:
                self._on_level(level(chunk))
            self._stream.write(chunk)
            if self._q.empty():
                self._busy.clear()
                if self._on_level:
                    self._on_level(0.0)

    def play(self, chunk: bytes) -> None:
        self._busy.set()
        self._q.put(chunk)

    def flush(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._busy.clear()

    @property
    def speaking(self) -> bool:
        return self._busy.is_set()

    def close(self) -> None:
        self.flush()
        self._q.put(None)
        self._stream.stop()
        self._stream.close()
