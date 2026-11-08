"""The AI core: a glass sphere with a holographic lattice, a scan line and LED "visor" eyes.

Drawn by the Orb. Everything is vector + fake bloom (layered translucent strokes), so it stays
sharp at any size and costs little to paint.

Eyes by state:  listening  LED bars that breathe with your voice and glance around
                speaking   happy LED arcs and a live voice waveform for a mouth
                thinking   a scanner light sweeping inside each visor
                sleeping   thin dim lines
Feelings:       love (LED hearts), sad (tilted dim bars), wink, surprised (rings)
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient


# -- helpers ---------------------------------------------------------------------------
def _alpha(c: QColor, a: int) -> QColor:
    c = QColor(c)
    c.setAlpha(max(0, min(255, int(a))))
    return c


def glow_stroke(p: QPainter, path: QPainterPath, color: QColor, width: float, strength: float = 1.0) -> None:
    """Draw a path with a soft bloom around it."""
    p.setBrush(Qt.BrushStyle.NoBrush)
    for mult, a in ((4.2, 22), (2.6, 45), (1.6, 90)):
        pen = QPen(_alpha(color, a * strength), width * mult)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawPath(path)
    core = QPen(_alpha(QColor(color).lighter(160), 255 * min(1.0, strength)), width)
    core.setCapStyle(Qt.PenCapStyle.RoundCap)
    core.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(core)
    p.drawPath(path)


def glow_fill(p: QPainter, path: QPainterPath, color: QColor, strength: float = 1.0) -> None:
    p.setBrush(Qt.BrushStyle.NoBrush)
    for w, a in ((10, 25), (6, 45), (3, 80)):
        pen = QPen(_alpha(color, a * strength), w)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawPath(path)
    p.setPen(Qt.PenStyle.NoPen)
    grad = QLinearGradient(path.boundingRect().topLeft(), path.boundingRect().bottomLeft())
    grad.setColorAt(0.0, _alpha(QColor(color).lighter(175), 255 * strength))
    grad.setColorAt(1.0, _alpha(QColor(color).lighter(120), 235 * strength))
    p.setBrush(grad)
    p.drawPath(path)


def heart_path(c: QPointF, size: float) -> QPainterPath:
    x, y, s = c.x(), c.y(), size
    path = QPainterPath()
    path.moveTo(x, y + s * 0.9)
    path.cubicTo(x - s * 1.4, y - s * 0.1, x - s * 0.6, y - s * 1.1, x, y - s * 0.35)
    path.cubicTo(x + s * 0.6, y - s * 1.1, x + s * 1.4, y - s * 0.1, x, y + s * 0.9)
    return path


def _bar(center: QPointF, w: float, h: float, angle: float = 0.0) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(QRectF(-w / 2, -h / 2, w, h), h / 2, h / 2)
    from PyQt6.QtGui import QTransform

    t = QTransform()
    t.translate(center.x(), center.y())
    t.rotate(angle)
    return t.map(path)


# -- the sphere ------------------------------------------------------------------------------
def draw_core(p: QPainter, orb, cx: float, cy: float, cr: float, color: QColor) -> None:
    t, level, flash = orb.t, orb.level, orb.flash_t
    center = QPointF(cx, cy)

    # outer bloom ring
    for w, a in ((14, 18), (8, 35), (3, 90)):
        p.setPen(QPen(_alpha(color, a + 60 * level + 80 * flash), w))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(center, cr, cr)

    # dark glass body with an energy glow inside
    body = QRadialGradient(QPointF(cx, cy + cr * 0.15), cr * 1.05)
    body.setColorAt(0.0, _alpha(QColor(color).darker(170), 245))
    body.setColorAt(0.55, _alpha(QColor(color).darker(330), 250))
    body.setColorAt(1.0, QColor(4, 8, 12, 255))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(body)
    p.drawEllipse(center, cr, cr)

    p.save()
    clip = QPainterPath()
    clip.addEllipse(center, cr * 0.985, cr * 0.985)
    p.setClipPath(clip)

    # energy pulse in the middle
    energy = QRadialGradient(center, cr * (0.55 + 0.35 * level))
    energy.setColorAt(0.0, _alpha(color, 70 + 120 * level + 120 * flash))
    energy.setColorAt(1.0, _alpha(color, 0))
    p.setBrush(energy)
    p.drawEllipse(center, cr, cr)

    # rotating hex lattice
    step = cr * 0.22
    ang = t * 0.15
    ca, sa = math.cos(ang), math.sin(ang)
    lattice = _alpha(color, 26 + 30 * level)
    p.setPen(QPen(lattice, 1))
    p.setBrush(Qt.BrushStyle.NoBrush)
    n = int(cr / step) + 2
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            hx = (i + (j % 2) * 0.5) * step
            hy = j * step * 0.866
            x = cx + hx * ca - hy * sa
            y = cy + hx * sa + hy * ca
            if (x - cx) ** 2 + (y - cy) ** 2 > cr * cr:
                continue
            hexagon = QPainterPath()
            for k in range(7):
                a = math.pi / 6 + k * math.pi / 3 + ang
                px, py = x + math.cos(a) * step * 0.5, y + math.sin(a) * step * 0.5
                hexagon.moveTo(px, py) if k == 0 else hexagon.lineTo(px, py)
            p.drawPath(hexagon)

    # scan line sweeping down
    sy = cy - cr + ((t * 0.45) % 1.0) * 2 * cr
    scan = QLinearGradient(QPointF(cx, sy - cr * 0.18), QPointF(cx, sy + cr * 0.02))
    scan.setColorAt(0.0, _alpha(color, 0))
    scan.setColorAt(0.9, _alpha(color, 60))
    scan.setColorAt(1.0, _alpha(QColor(color).lighter(150), 80))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(scan)
    p.drawRect(QRectF(cx - cr, sy - cr * 0.18, cr * 2, cr * 0.2))
    p.restore()

    # segmented inner ring, rotating
    seg_r = cr * 0.9
    p.setBrush(Qt.BrushStyle.NoBrush)
    for k in range(12):
        a0 = (k * 30 + math.degrees(t * 0.6)) % 360
        pen = QPen(_alpha(color, 70 + 90 * level if k % 3 else 160), 2)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        p.setPen(pen)
        p.drawArc(QRectF(cx - seg_r, cy - seg_r, seg_r * 2, seg_r * 2), int(a0 * 16), int(18 * 16))

    # glass highlight
    hl = QRadialGradient(QPointF(cx - cr * 0.35, cy - cr * 0.5), cr * 0.45)
    hl.setColorAt(0.0, QColor(255, 255, 255, 46))
    hl.setColorAt(1.0, QColor(255, 255, 255, 0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(hl)
    p.drawEllipse(QPointF(cx - cr * 0.35, cy - cr * 0.5), cr * 0.45, cr * 0.3)


# -- the face ------------------------------------------------------------------------------
def draw_face(p: QPainter, orb, cx: float, cy: float, cr: float, color: QColor, feeling: str,
              blinking: bool) -> None:
    state, t, level = orb.state, orb.t, orb.level
    eye_c = QColor(color).lighter(135)
    ex, ey = cr * 0.33, cy - cr * 0.07
    ew, eh = cr * 0.38, cr * 0.15
    look = QPointF(orb.look.x() * cr * 0.07, orb.look.y() * cr * 0.06)
    lw = max(2.0, cr * 0.045)  # line width
    mouth_y = cy + cr * 0.36

    if feeling == "love":
        pink = QColor(255, 80, 150)
        beat = 1 + 0.14 * max(0.0, math.sin(t * 9))
        for side in (-1, 1):
            glow_fill(p, heart_path(QPointF(cx + side * ex, ey), cr * 0.13 * beat), pink)
        _smile(p, cx, mouth_y, cr, pink, lw)
        return
    if feeling == "sad":
        blue = QColor(110, 170, 255)
        for side in (-1, 1):
            glow_fill(p, _bar(QPointF(cx + side * ex, ey + eh * 0.4), ew * 0.85, eh * 0.75, side * 14), blue, 0.75)
        arc = QPainterPath()
        arc.moveTo(cx - cr * 0.16, mouth_y + cr * 0.05)
        arc.quadTo(cx, mouth_y - cr * 0.08, cx + cr * 0.16, mouth_y + cr * 0.05)
        glow_stroke(p, arc, blue, lw, 0.8)
        drop = (t * 0.7) % 1.0
        tear = QPainterPath()
        tear.addEllipse(QPointF(cx + ex * 1.05, ey + eh + drop * cr * 0.35), lw * 0.9, lw * 1.3)
        glow_fill(p, tear, blue, 1 - drop)
        return
    if feeling == "surprised":
        for side in (-1, 1):
            ring = QPainterPath()
            ring.addEllipse(QPointF(cx + side * ex, ey), cr * 0.12, cr * 0.12)
            glow_stroke(p, ring, eye_c, lw * 1.1)
        o = QPainterPath()
        o.addEllipse(QPointF(cx, mouth_y + cr * 0.02), cr * 0.06, cr * 0.075)
        glow_stroke(p, o, eye_c, lw * 0.9)
        return
    if feeling == "wink":
        glow_fill(p, _bar(QPointF(cx - ex, ey) + look, ew, eh), eye_c)
        _arc_eye(p, QPointF(cx + ex, ey), ew, eh, eye_c, lw)
        _smile(p, cx, mouth_y, cr, eye_c, lw)
        return

    # -- states --
    if state in {"sleeping", "offline", "connecting"} or blinking:
        dim = 0.35 if state in {"sleeping", "offline"} else 0.7
        for side in (-1, 1):
            line = QPainterPath()
            c = QPointF(cx + side * ex, ey + eh * 0.2)
            line.moveTo(c.x() - ew / 2, c.y())
            line.lineTo(c.x() + ew / 2, c.y())
            glow_stroke(p, line, eye_c, lw * 0.8, dim)
        if state in {"sleeping", "offline"}:
            _flat_mouth(p, cx, mouth_y, cr * 0.08, eye_c, lw * 0.7, 0.3)
        return

    if state == "speaking" and level > 0.18:
        for side in (-1, 1):
            _arc_eye(p, QPointF(cx + side * ex, ey), ew, eh, eye_c, lw)
    elif state == "thinking":
        for k, side in enumerate((-1, 1)):
            c = QPointF(cx + side * ex, ey)
            visor = _bar(c, ew, eh)
            glow_stroke(p, visor, eye_c, lw * 0.5, 0.55)
            pos = math.sin(t * 4.2 + k * 0.6)  # scanner light sweeping left-right
            dot = QPainterPath()
            dot.addEllipse(QPointF(c.x() + pos * (ew / 2 - eh / 2), c.y()), eh * 0.42, eh * 0.42)
            glow_fill(p, dot, QColor("#ffb547"))
    else:
        grow = 1 + level * 0.7 if state == "listening" else 1.0
        for side in (-1, 1):
            c = QPointF(cx + side * ex, ey) + look
            glow_fill(p, _bar(c, ew, eh * grow), eye_c)
            # LED segments
            p.setPen(QPen(_alpha(QColor(4, 8, 12), 110), max(1.0, cr * 0.012)))
            seg = ew / 7
            for k in range(1, 7):
                x = c.x() - ew / 2 + seg * k
                p.drawLine(QPointF(x, c.y() - eh * grow / 2 + 1), QPointF(x, c.y() + eh * grow / 2 - 1))

    # mouth: a live voice waveform
    width = cr * 0.42
    if state == "speaking":
        amp = cr * (0.02 + 0.12 * level)
    elif state == "listening":
        amp = cr * 0.06 * level
    else:
        amp = 0.0
    if state == "muted":
        _flat_mouth(p, cx, mouth_y, width * 0.35, QColor("#ff5c6c"), lw * 0.8, 0.8)
        return
    wave = QPainterPath()
    steps = 28
    for i in range(steps + 1):
        u = i / steps
        x = cx - width / 2 + u * width
        env = math.sin(math.pi * u)  # zero at the ends
        y = mouth_y + amp * env * math.sin(u * 14 + t * 18) * (0.6 + 0.4 * math.sin(t * 5 + u * 6))
        wave.moveTo(x, y) if i == 0 else wave.lineTo(x, y)
    glow_stroke(p, wave, eye_c, lw * 0.75, 0.55 + 0.45 * min(1.0, level * 2 + 0.3))


def _arc_eye(p, c: QPointF, ew: float, eh: float, color: QColor, lw: float) -> None:
    arc = QPainterPath()
    arc.moveTo(c.x() - ew / 2, c.y() + eh * 0.55)
    arc.quadTo(c.x(), c.y() - eh * 1.6, c.x() + ew / 2, c.y() + eh * 0.55)
    glow_stroke(p, arc, color, lw * 1.2)


def _smile(p, cx: float, my: float, cr: float, color: QColor, lw: float) -> None:
    arc = QPainterPath()
    arc.moveTo(cx - cr * 0.2, my - cr * 0.03)
    arc.quadTo(cx, my + cr * 0.16, cx + cr * 0.2, my - cr * 0.03)
    glow_stroke(p, arc, color, lw)


def _flat_mouth(p, cx: float, my: float, half: float, color: QColor, lw: float, strength: float) -> None:
    line = QPainterPath()
    line.moveTo(cx - half, my)
    line.lineTo(cx + half, my)
    glow_stroke(p, line, color, lw, strength)


# -- neural ring --------------------------------------------------------------------------
def draw_neural_ring(p: QPainter, orb, cx: float, cy: float, r: float, color: QColor) -> None:
    """Rotating constellation of nodes around the core; they light up with sound."""
    n = 24
    t = orb.t
    pts = []
    for k in range(n):
        a = math.tau * k / n + t * 0.12
        wob = 1 + 0.035 * math.sin(t * 1.3 + k * 1.7)
        rr = r * 1.24 * wob
        pts.append((QPointF(cx + math.cos(a) * rr, cy + math.sin(a) * rr), orb.bars[k * 3 % len(orb.bars)]))
    for k, (pt, v) in enumerate(pts):
        nxt = pts[(k + 1) % n][0]
        skip = pts[(k + 3) % n][0]
        p.setPen(QPen(_alpha(color, 35 + 140 * v), 1))
        p.drawLine(pt, nxt)
        if k % 2 == 0:
            p.setPen(QPen(_alpha(color, 14 + 80 * v), 1))
            p.drawLine(pt, skip)
    p.setPen(Qt.PenStyle.NoPen)
    for pt, v in pts:
        size = 1.8 + 3.2 * v
        p.setBrush(_alpha(color, 50 + 60 * v))
        p.drawEllipse(pt, size * 2.2, size * 2.2)
        p.setBrush(_alpha(QColor(color).lighter(160), 170 + 85 * v))
        p.drawEllipse(pt, size, size)
