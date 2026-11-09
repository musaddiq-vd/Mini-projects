"""Loaded at start-up of the packaged app (as sitecustomize): openwakeword imports its training helper (which needs scipy and
scikit-learn, ~165 MB) as soon as it loads. Jarvis only detects wake words and never trains them, so a small
stand-in module is used instead of shipping those libraries."""

import sys
import types

_stub = types.ModuleType("openwakeword.custom_verifier_model")


def train_custom_verifier(*args, **kwargs):
    raise RuntimeError("Training custom wake-word verifiers isn't included in the installed app.")


_stub.train_custom_verifier = train_custom_verifier
sys.modules.setdefault("openwakeword.custom_verifier_model", _stub)
