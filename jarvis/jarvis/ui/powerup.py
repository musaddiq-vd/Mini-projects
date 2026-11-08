"""The power-up animation when Jarvis wakes: energy gathers into the core, then bursts out.

Timeline (t goes 0 -> 1 over DURATION seconds):
  0.00-0.30  charge: sparks are pulled into the core, which shrinks and glows hotter
  0.30       burst: white flash, three shockwave rings, lightning bolts, sparks fly out
  0.30-1.00  settle: a fast spinning segmented ring slows down and fades, the core pops then relaxes
"""

from __future__ import annotations

import math
import random

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainterPath, QPen, QRadialGradient

DURATION = 1.7
BURST = 0.30


def new_state() -> dict:
    rnd = random.Random()
    return {
        "t": 0.0,
        "in": [(rnd.uniform(0, 2 * math.pi), rnd.uniform(1.6, 2.6), rnd.uniform(0.7, 1.0)) for _ in range(46)],
        "out": [(rnd.uniform(0, 2 * math.pi), rnd.uniform(1.4, 3.0), rnd.uniform(1.5, 3.5)) for _ in range(60)],
        "bolts": [_bolt(rnd, rnd.uniform(0, 2 * math.pi)) for _ in range(7)],
    }


def _bolt(rnd: random.Random, angle: float) -> list[tuple[float, float]]:
    """A jagged lightning line as (radius factor, angle) points from the core outwards."""
    pts, a = [], angle
    for i in range(7):
        a += rnd.uniform(-0.16, 0.16)
        pts.append((0.55 + i * 0.28, a))
    return pts


def core_scale(t: float) -> float:
    """How big the core is during the power-up (1 = normal)."""
    if t <= 0 or t >= 1:
        return 1.0
    if t < BURST:  # charge: shrink a little
        return 1 - 0.14 * (t / BURST)
    k = (t - BURST) / (1 - BURST)  # pop out, then an elastic settle
    return 1 + 0.32 * math.exp(-5 * k) * math.cos(9 * k)


def glow_boost(t: float) -> float:
    if t <= 0 or t >= 1:
        return 0.0
    if t < BURST:
        return 0.6 * (t / BURST)
    return 1.0 * math.exp(-4 * (t - BURST) / (1 - BURST))


def draw(p, state: dict, cx: float, cy: float, r: float, color: QColor) -> None:
    t = state["t"]
    if t <= 0 or t >= 1:
        return
    p.save()
    c = QPointF(cx, cy)
    white = QColor("#ffffff")
    if t < BURST:  # -- charge
        k = t / BURST
        for ang, dist, size in state["in"]:
            d = r * (dist - (dist - 0.45) * (k ** 1.6))
            pos = QPointF(cx + math.cos(ang + k * 1.2) * d, cy + math.sin(ang + k * 1.2) * d)
            col = QColor(color)
            col.setAlphaF(min(1.0, 0.25 + k))
            p.setPen(QPen(col, 1.3 * size))
            tail = QPointF(cx + math.cos(ang + k * 1.2 - 0.12) * (d + r * 0.25),
                           cy + math.sin(ang + k * 1.2 - 0.12) * (d + r * 0.25))
            p.drawLine(pos, tail)
        halo = QRadialGradient(c, r * 1.1)
        hot = QColor(color).lighter(160)
        hot.setAlphaF(0.55 * k)
        halo.setColorAt(0, hot)
        halo.setColorAt(1, QColor(0, 0, 0, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(halo)
        p.drawEllipse(c, r * 1.1, r * 1.1)
    else:  # -- burst and settle
        k = (t - BURST) / (1 - BURST)
        if k < 0.25:  # the flash
            flash = QRadialGradient(c, r * 3.2)
            fw = QColor(white)
            fw.setAlphaF(0.75 * (1 - k / 0.25))
            flash.setColorAt(0, fw)
            fw2 = QColor(color)
            fw2.setAlphaF(0.35 * (1 - k / 0.25))
            flash.setColorAt(0.45, fw2)
            flash.setColorAt(1, QColor(0, 0, 0, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(flash)
            p.drawEllipse(c, r * 3.2, r * 3.2)
        for i in range(3):  # shockwaves
            kk = k * 1.25 - i * 0.14
            if 0 < kk < 1:
                ring = QColor(white if i == 0 else color)
                ring.setAlphaF(0.85 * (1 - kk) ** 1.4)
                p.setPen(QPen(ring, max(0.6, 5 * (1 - kk))))
                p.setBrush(Qt.BrushStyle.NoBrush)
                rr = r * (0.7 + 2.3 * (1 - (1 - kk) ** 2.2))
                p.drawEllipse(c, rr, rr)
        if k < 0.45:  # lightning
            alpha = 1 - k / 0.45
            for bolt in state["bolts"]:
                path = QPainterPath()
                for j, (rf, ang) in enumerate(bolt):
                    pt = QPointF(cx + math.cos(ang) * r * rf, cy + math.sin(ang) * r * rf)
                    path.moveTo(pt) if j == 0 else path.lineTo(pt)
                glow = QColor(color)
                glow.setAlphaF(0.35 * alpha)
                p.setPen(QPen(glow, 6, cap=Qt.PenCapStyle.RoundCap))
                p.drawPath(path)
                core = QColor(white)
                core.setAlphaF(0.95 * alpha)
                p.setPen(QPen(core, 1.6, cap=Qt.PenCapStyle.RoundCap))
                p.drawPath(path)
        for ang, dist, speed in state["out"]:  # sparks
            d = r * (0.6 + dist * (1 - (1 - k) ** 2.5))
            if d > r * 3.4:
                continue
            col = QColor(color if speed < 2.6 else white)
            col.setAlphaF(max(0.0, 1 - k) ** 1.2)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col)
            size = 1.2 + speed * 0.7 * (1 - k)
            p.drawEllipse(QPointF(cx + math.cos(ang) * d, cy + math.sin(ang) * d), size, size)
        # the segmented ring: spins fast, slows down, fades
        spin = 720 * (1 - (1 - k) ** 3)
        ring_col = QColor(color)
        ring_col.setAlphaF(0.9 * (1 - k))
        p.setPen(QPen(ring_col, 3, cap=Qt.PenCapStyle.FlatCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        rr = r * 1.32
        box = QRectF(cx - rr, cy - rr, 2 * rr, 2 * rr)
        for seg in range(6):
            p.drawArc(box, int((spin + seg * 60) * 16), int(38 * 16))
    p.restore()


def text_alpha(t: float) -> float:
    """'ONLINE' label under the orb, shown during the settle."""
    if t <= BURST or t >= 1:
        return 0.0
    k = (t - BURST) / (1 - BURST)
    return min(1.0, k * 4) * (1 - k) ** 0.6
