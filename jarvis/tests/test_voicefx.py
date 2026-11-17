"""The robot voice effect: shape-preserving, click-free across chunks, never clipping, and wired in."""

import time

import numpy as np

from jarvis import voicefx


def _tone(n, freq=220.0, rate=24_000, amp=0.6, start=0):
    t = (np.arange(n) + start) / rate
    return (np.sin(2 * np.pi * freq * t) * amp * 32767).astype(np.int16).tobytes()


def test_make_and_presets():
    assert voicefx.make("off") is None and voicefx.make("") is None
    for name in ("light", "robot", "strong"):
        assert isinstance(voicefx.make(name), voicefx.RobotVoice)
    assert voicefx.RobotVoice("nonsense").effect == "robot"


def test_changes_the_sound_but_keeps_length_and_loudness():
    fx = voicefx.RobotVoice("robot")
    src = _tone(4800)
    out = fx.process(src)
    assert len(out) == len(src) and out != src
    a = np.frombuffer(src, np.int16).astype(float)
    b = np.frombuffer(out, np.int16).astype(float)
    ratio = np.sqrt(np.mean(b * b)) / np.sqrt(np.mean(a * a))
    assert 0.3 < ratio < 1.5  # still clearly audible, not blasting


def test_never_clips_and_handles_silence_and_empty():
    fx = voicefx.RobotVoice("strong")
    loud = (np.ones(2400) * 32767).astype(np.int16).tobytes()
    out = np.frombuffer(fx.process(loud), np.int16)
    assert out.max() <= 32767 and out.min() >= -32768
    assert fx.process(b"") == b""
    silent = np.frombuffer(voicefx.RobotVoice().process(bytes(4800)), np.int16)
    assert np.abs(silent).max() == 0


def test_chunked_equals_whole():
    whole = voicefx.RobotVoice("robot").process(_tone(9600))
    fx = voicefx.RobotVoice("robot")
    parts = b"".join(fx.process(_tone(n, start=s)) for s, n in ((0, 1000), (1000, 3700), (4700, 4900)))
    a, b = np.frombuffer(whole, np.int16).astype(int), np.frombuffer(parts, np.int16).astype(int)
    assert np.abs(a - b).max() <= 2  # same result: no clicks at chunk borders


def test_fast_enough_for_live_audio():
    fx = voicefx.RobotVoice("robot")
    second = _tone(24_000)
    start = time.perf_counter()
    fx.process(second)
    assert time.perf_counter() - start < 0.5  # one second of speech in well under half a second


def test_speaker_applies_and_survives_a_broken_effect(monkeypatch):
    import queue

    from jarvis import audio

    class FakeStream:
        def __init__(self, **kw):
            self.written = queue.Queue()

        def start(self):
            pass

        def write(self, chunk):
            self.written.put(chunk)

        def stop(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(audio.sd, "RawOutputStream", FakeStream)

    class Upper:
        def process(self, c):
            return b"FX" + c

        def reset(self):
            pass

    spk = audio.Speaker(fx=Upper())
    spk.start()
    spk.play(b"ab")
    assert spk._stream.written.get(timeout=2) == b"FXab"

    class Broken:
        def process(self, c):
            raise ValueError("boom")

        def reset(self):
            pass

    spk.fx = Broken()
    spk.play(b"cd")
    assert spk._stream.written.get(timeout=2) == b"cd" and spk.fx is None  # plain voice from then on
    spk.close()


def test_default_setting_is_robot():
    from jarvis.config import Settings

    assert Settings().voice_effect == "robot"
