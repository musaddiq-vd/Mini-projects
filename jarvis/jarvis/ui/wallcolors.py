"""Colours from the desktop wallpaper, for a theme that matches it.

palette(image) finds the two most vivid, distinct colours (weighted by how much of the picture they cover)
and tunes them to glow on the dark UI and stay readable on the light one. A grey or black-and-white wallpaper
gives a calm grey-blue instead of a random tint.
"""

from __future__ import annotations

import colorsys
import ctypes
import os
import platform
from pathlib import Path

FALLBACK = ("#9fb3c4", "#5d7385")
SPI_GETDESKWALLPAPER = 0x0073


def wallpaper_path() -> Path | None:
    """The current Windows wallpaper file, or None (solid colour, slideshow not cached, other OS)."""
    if platform.system() != "Windows":
        return None
    buf = ctypes.create_unicode_buffer(520)
    try:
        if not ctypes.windll.user32.SystemParametersInfoW(SPI_GETDESKWALLPAPER, 520, buf, 0):
            return None
    except (AttributeError, OSError):
        return None
    path = Path(buf.value) if buf.value else None
    if path and path.is_file():
        return path
    # Windows keeps a copy of the current wallpaper here (also for slideshows and Spotlight)
    cached = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Themes" / "TranscodedWallpaper"
    return cached if cached.is_file() else None


def signature(path: Path | None) -> str:
    """Changes whenever the wallpaper changes (path, size or time)."""
    if path is None:
        return ""
    try:
        st = path.stat()
    except OSError:
        return ""
    return f"{path}|{st.st_size}|{int(st.st_mtime)}"


def _tune(rgb: tuple[float, float, float], light_min: float) -> str:
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    s = max(s, 0.55)
    v = max(v, light_min)
    r, g, b = colorsys.hsv_to_rgb(h, min(1.0, s), min(1.0, v))
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


def palette(image) -> tuple[str, str]:
    """(accent, second) from a PIL image."""
    img = image.convert("RGB")
    img.thumbnail((96, 96))
    pixels = list(img.get_flattened_data() if hasattr(img, "get_flattened_data") else img.getdata())
    bins: dict[int, list[float]] = {}  # hue bin -> [weight, r, g, b]
    for r, g, b in pixels:
        rf, gf, bf = r / 255, g / 255, b / 255
        h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
        if s < 0.3 or v < 0.3:
            continue  # greys, near-black and washed-out pixels don't make a theme
        weight = s * v
        k = int(h * 24) % 24
        cell = bins.setdefault(k, [0.0, 0.0, 0.0, 0.0])
        cell[0] += weight
        cell[1] += rf * weight
        cell[2] += gf * weight
        cell[3] += bf * weight
    total = sum(c[0] for c in bins.values())
    if not bins or total < len(pixels) * 0.03:  # almost no colour in the picture
        return FALLBACK
    ranked = sorted(bins.items(), key=lambda kv: -kv[1][0])
    first_bin, first = ranked[0]
    accent_rgb = (first[1] / first[0], first[2] / first[0], first[3] / first[0])
    second_rgb = None
    for k, cell in ranked[1:]:
        distance = min(abs(k - first_bin), 24 - abs(k - first_bin))
        if distance >= 3 and cell[0] >= first[0] * 0.08:  # at least ~45° away and not a speck
            second_rgb = (cell[1] / cell[0], cell[2] / cell[0], cell[3] / cell[0])
            break
    if second_rgb is None:  # one-colour wallpaper: a neighbour of that colour
        h, s, v = colorsys.rgb_to_hsv(*accent_rgb)
        second_rgb = colorsys.hsv_to_rgb((h + 0.11) % 1.0, s, v)
    return _tune(accent_rgb, 0.85), _tune(second_rgb, 0.7)


def from_wallpaper() -> tuple[str, str] | None:
    path = wallpaper_path()
    if path is None:
        return None
    try:
        from PIL import Image

        with Image.open(path) as img:
            return palette(img)
    except Exception:
        return None
