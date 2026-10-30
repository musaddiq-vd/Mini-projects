"""Network helper.

Many home networks advertise IPv6 but can't actually route it. Python then
tries every IPv6 address first and each one times out, so connecting to
Gemini takes minutes or fails. Putting IPv4 addresses first fixes that while
keeping IPv6 as a fallback.
"""

from __future__ import annotations

import socket

_original_getaddrinfo = socket.getaddrinfo


def _ipv4_first(*args, **kwargs):
    results = _original_getaddrinfo(*args, **kwargs)
    return sorted(results, key=lambda r: r[0] != socket.AF_INET)


def prefer_ipv4() -> None:
    socket.getaddrinfo = _ipv4_first
