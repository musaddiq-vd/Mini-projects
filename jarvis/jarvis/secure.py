"""Encrypt secrets at rest (the Gemini API key) with Windows DPAPI.

DPAPI ties the encryption to your Windows user account: another user, another PC or a copied config file
can't decrypt it. On other systems the value is stored as before (the file is still only in your home).
"""

from __future__ import annotations

import base64
import ctypes
import logging
import platform

PREFIX = "dpapi:"
log = logging.getLogger("jarvis.secure")
_ENTROPY = b"jarvis-assistant-v1"


def available() -> bool:
    return platform.system() == "Windows"


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_uint32), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _blob(data: bytes) -> _Blob:
    buf = ctypes.create_string_buffer(data, len(data))
    return _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))


def _call(fn_name: str, data: bytes) -> bytes:
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    src, entropy, out = _blob(data), _blob(_ENTROPY), _Blob()
    fn = getattr(crypt32, fn_name)
    ok = fn(ctypes.byref(src), None, ctypes.byref(entropy), None, None, 0x01, ctypes.byref(out))  # UI_FORBIDDEN
    if not ok:
        raise OSError(f"{fn_name} failed ({ctypes.GetLastError()})")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(out.pbData)


def protect(text: str) -> str:
    """'abc' -> 'dpapi:...' (unchanged when empty, already protected or DPAPI isn't available)."""
    if not text or text.startswith(PREFIX) or not available():
        return text
    try:
        return PREFIX + base64.b64encode(_call("CryptProtectData", text.encode("utf-8"))).decode("ascii")
    except OSError as exc:
        log.warning("couldn't encrypt a secret: %s", exc)
        return text


def unprotect(text: str) -> str:
    """'dpapi:...' -> 'abc'. A value from another account/PC gives '' (the user enters it again)."""
    if not text or not text.startswith(PREFIX):
        return text
    if not available():
        return ""
    try:
        return _call("CryptUnprotectData", base64.b64decode(text[len(PREFIX):])).decode("utf-8")
    except (OSError, ValueError) as exc:
        log.warning("couldn't decrypt the saved API key (%s) - please enter it again in Settings", exc)
        return ""
