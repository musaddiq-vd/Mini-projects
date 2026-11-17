"""Voice effects for Jarvis's spoken replies: a clear "robot" sound, like a movie AI.

The effect works on the 24 kHz 16-bit mono PCM chunks the live model sends, keeps its state between chunks
(so there are no clicks at chunk borders) and stays easy to understand:
- ring modulation with a low carrier gives the metallic, synthetic tone
- a short comb filter (a few milliseconds of feedback delay) adds the "inside a machine" resonance
- a gentle high-pass keeps the voice crisp and removes boom
Levels: "off", "light", "robot" (default) and "strong".
"""

from __future__ import annotations

import numpy as np

EFFECTS = ("off", "light", "robot", "strong")
# effect -> (ring-mod carrier Hz, ring-mod mix, comb delay ms, comb feedback, comb mix)
PRESETS = {
    "light": (38.0, 0.25, 4.0, 0.35, 0.25),
    "robot": (52.0, 0.45, 5.5, 0.55, 0.35),
    "strong": (70.0, 0.70, 7.0, 0.70, 0.45),
}


class RobotVoice:
    def __init__(self, effect: str = "robot", rate: int = 24_000):
        self.effect = effect if effect in PRESETS else "robot"
        self.rate = rate
        carrier, self.ring_mix, delay_ms, self.feedback, self.comb_mix = PRESETS[self.effect]
        self._step = 2 * np.pi * carrier / rate
        self._phase = 0.0
        self._delay = max(1, int(rate * delay_ms / 1000))
        self._buf = np.zeros(self._delay, dtype=np.float32)  # the comb filter's last output samples
        self._hp_prev_in = 0.0
        self._hp_prev_out = 0.0
        self._hp_a = float(np.exp(-2 * np.pi * 120.0 / rate))  # one-pole high-pass at ~120 Hz

    def reset(self) -> None:
        """Forget the tail of the last reply (after an interruption)."""
        self._buf[:] = 0
        self._hp_prev_in = self._hp_prev_out = 0.0

    def process(self, pcm16: bytes) -> bytes:
        if not pcm16:
            return pcm16
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        n = len(x)
        # ring modulation (phase kept between chunks)
        t = self._phase + self._step * np.arange(n, dtype=np.float64)
        self._phase = float((self._phase + self._step * n) % (2 * np.pi))
        ring = x * np.sin(t).astype(np.float32)
        y = (1 - self.ring_mix) * x + self.ring_mix * ring * 1.6
        # comb filter with feedback: out[i] = y[i] + fb * out[i - d]
        d = self._delay
        out = np.empty(n + d, dtype=np.float32)
        out[:d] = self._buf
        for start in range(0, n, d):  # block-wise: each block only needs the previous block's output
            end = min(start + d, n)
            out[d + start:d + end] = y[start:end] + self.feedback * out[start:end]
        comb = out[d:]
        self._buf = out[-d:].copy()
        y = (1 - self.comb_mix) * y + self.comb_mix * comb * (1 - self.feedback)
        # one-pole high-pass (sample by sample state only at the edges: vectorised with a short loop-free trick)
        hp = np.empty(n, dtype=np.float32)
        a, prev_in, prev_out = self._hp_a, self._hp_prev_in, self._hp_prev_out
        for i in range(n):
            prev_out = a * (prev_out + y[i] - prev_in)
            prev_in = float(y[i])
            hp[i] = prev_out
        self._hp_prev_in, self._hp_prev_out = prev_in, prev_out
        # soft limiter so peaks never clip
        hp = np.tanh(hp * 1.4) / np.tanh(1.4)
        return (np.clip(hp, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def make(effect: str | None):
    """A chunk -> chunk function for the speaker, or None for the plain voice."""
    if not effect or effect == "off":
        return None
    fx = RobotVoice(effect)
    return fx
