"""Optional local wake word using openwakeword (pip install openwakeword).

Runs fully offline. While asleep no audio is sent anywhere.
Built-in model names include: hey_jarvis, alexa, hey_mycroft, hey_rhasspy.
"""

from __future__ import annotations

import numpy as np


class WakeWord:
    def __init__(self, model_name: str, threshold: float = 0.5):
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError as exc:
            raise RuntimeError("wake word needs: pip install openwakeword==0.6.0 (inside .venv)") from exc
        openwakeword.utils.download_models([model_name])
        self.model = Model(wakeword_models=[model_name], inference_framework="onnx")
        self.threshold = threshold

    def heard(self, pcm16: bytes) -> bool:
        scores = self.model.predict(np.frombuffer(pcm16, dtype=np.int16))
        if max(scores.values(), default=0.0) >= self.threshold:
            self.model.reset()
            return True
        return False
