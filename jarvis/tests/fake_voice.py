"""Synthetic voices for tests: a pulse train through formant filters (no real recording)."""

from __future__ import annotations

import numpy as np

RATE = 16_000
VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (530, 1840, 2480), (300, 870, 2240), (660, 1720, 2410)]


def _resonate(x: np.ndarray, freq: float, bw: float) -> np.ndarray:
    r = np.exp(-np.pi * bw / RATE)
    a1, a2 = -2 * r * np.cos(2 * np.pi * freq / RATE), r * r
    y = np.zeros_like(x)
    y1 = y2 = 0.0
    g = 1 - r
    for i in range(len(x)):
        v = g * x[i] - a1 * y1 - a2 * y2
        y[i] = v
        y2, y1 = y1, v
    return y


def voice(f0: float, tract: float, seconds: float = 3.0, seed: int = 0, tilt: float = 0.0,
          pause: bool = True) -> np.ndarray:
    """A "sentence" by a speaker with pitch f0 and vocal tract scale `tract` (formants * tract)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    out = np.zeros(n)
    seg = int(0.25 * RATE)
    for start in range(0, n, seg):
        f = f0 * (1 + rng.normal(0, 0.04))
        period = int(RATE / f)
        src = np.zeros(seg)
        src[::period] = 1.0
        src += rng.normal(0, 0.02 + tilt, seg)
        formants = VOWELS[rng.integers(len(VOWELS))]
        y = sum(_resonate(src, fr * tract, 80 + 40 * k) for k, fr in enumerate(formants))
        if pause and rng.random() < 0.15:
            y *= 0.0
        out[start:start + seg] = y[: n - start]
    out = out / (np.max(np.abs(out)) + 1e-9) * 0.5
    return (out * 32767).astype(np.int16)
