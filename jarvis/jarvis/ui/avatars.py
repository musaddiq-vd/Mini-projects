"""Avatar packs for the orb's core: robot, cat, anime and minimal (the default AI core lives in aiface.py).

Every avatar reacts the same way: blinks, closes its eyes while asleep, looks up while thinking, talks with
the voice level, shows heart eyes for love and squints happily for joy.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient

from jarvis.ui.aiface import heart_path

AVATARS = {
    "ai": "AI core",
    "robot": "Robot",
    "cat": "Cat",
    "anime": "Anime",
    "minimal": "Minimal",
}


def _a(c: QColor, alpha: float) -> QColor:
    out = QColor(c)
    out.setAlphaF(max(0.0, min(1.0, alpha)))
    return out


def _mood(orb, feeling: str) -> str:
    """One word the drawings use: sleep, think, love, happy, sad, talk or idle."""
    if orb.state in ("sleeping", "offline"):
        return "sleep"
    if feeling in ("love",):
        return "love"
    if feeling in ("happy", "wink", "star"):
        return "happy"
    if feeling in ("sad",):
        return "sad"
    if orb.state == "thinking":
        return "think"
    if orb.state == "speaking":
        return "talk"
    return "idle"


def draw(p: QPainter, orb, kind: str, cx: float, cy: float, cr: float, color: QColor, feeling: str,
         blinking: bool) -> None:
    fn = {"robot": _robot, "cat": _cat, "anime": _anime, "minimal": _minimal}.get(kind)
    if fn is None:
        return
    p.save()
    fn(p, orb, cx, cy, cr, color, _mood(orb, feeling), blinking)
    p.restore()


# -- shared pieces ------------------------------------------------------------------------------
def _closed_eye(p, c: QPointF, w: float, color: QColor, lw: float, happy: bool = False) -> None:
    path = QPainterPath()
    path.moveTo(c.x() - w / 2, c.y())
    if happy:  # ^  ^
        path.quadTo(c.x(), c.y() - w * 0.55, c.x() + w / 2, c.y())
    else:  # ‿  ‿
        path.quadTo(c.x(), c.y() + w * 0.35, c.x() + w / 2, c.y())
    p.setPen(QPen(color, lw, cap=Qt.PenCapStyle.RoundCap))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)


def _heart_eye(p, c: QPointF, size: float) -> None:
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#ff4f8b"))
    p.drawPath(heart_path(c, size))


def _talk_open(orb) -> float:
    return min(1.0, orb.level * 2.2) if orb.state == "speaking" else 0.0


def _look(orb, cr: float, kx: float = 0.08, ky: float = 0.06) -> QPointF:
    return QPointF(orb.look.x() * cr * kx, orb.look.y() * cr * ky)


# -- robot --------------------------------------------------------------------------------------
def _robot(p, orb, cx, cy, cr, color, mood, blinking) -> None:
    head = QRectF(cx - cr * 0.92, cy - cr * 0.78, cr * 1.84, cr * 1.62)
    # antenna with a blinking light
    p.setPen(QPen(_a(color.lighter(110), 0.9), cr * 0.05, cap=Qt.PenCapStyle.RoundCap))
    p.drawLine(QPointF(cx, head.top()), QPointF(cx, head.top() - cr * 0.28))
    on = (math.sin(orb.t * 4) > 0) or mood == "talk"
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_a(QColor("#ff5c6c") if mood == "sad" else color.lighter(150), 1.0 if on else 0.35))
    p.drawEllipse(QPointF(cx, head.top() - cr * 0.32), cr * 0.08, cr * 0.08)
    # head plate
    grad = QLinearGradient(head.topLeft(), head.bottomLeft())
    grad.setColorAt(0, QColor("#2b3a47"))
    grad.setColorAt(1, QColor("#121a22"))
    p.setBrush(grad)
    p.setPen(QPen(_a(color, 0.85), cr * 0.045))
    p.drawRoundedRect(head, cr * 0.32, cr * 0.32)
    # ear bolts
    p.setBrush(_a(color, 0.6))
    p.setPen(Qt.PenStyle.NoPen)
    for side in (-1, 1):
        p.drawRoundedRect(QRectF(cx + side * cr * 0.92 - cr * 0.06, cy - cr * 0.2, cr * 0.12, cr * 0.36),
                          cr * 0.04, cr * 0.04)
    # visor
    visor = QRectF(cx - cr * 0.7, cy - cr * 0.42, cr * 1.4, cr * 0.5)
    p.setBrush(QColor(4, 10, 16, 235))
    p.drawRoundedRect(visor, cr * 0.18, cr * 0.18)
    look = _look(orb, cr)
    ex, ey = cr * 0.32, visor.center().y() + look.y()
    eye_c = color.lighter(140)
    for side in (-1, 1):
        c = QPointF(cx + side * ex + look.x(), ey)
        if mood == "love":
            _heart_eye(p, c, cr * 0.2)
        elif mood == "sleep" or blinking:
            p.setPen(QPen(eye_c, cr * 0.05, cap=Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(c.x() - cr * 0.12, c.y()), QPointF(c.x() + cr * 0.12, c.y()))
        elif mood == "happy":
            _closed_eye(p, c, cr * 0.26, eye_c, cr * 0.05, happy=True)
        else:
            h = cr * (0.2 if mood != "think" else 0.12)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_a(eye_c, 0.25))
            p.drawRoundedRect(QRectF(c.x() - cr * 0.15, c.y() - h * 0.75, cr * 0.3, h * 1.5), cr * 0.05, cr * 0.05)
            p.setBrush(eye_c)
            p.drawRoundedRect(QRectF(c.x() - cr * 0.1, c.y() - h / 2, cr * 0.2, h), cr * 0.04, cr * 0.04)
    # speaker grill mouth
    open_ = _talk_open(orb)
    my = cy + cr * 0.42
    bars = 7
    for i in range(bars):
        x = cx - cr * 0.36 + i * cr * 0.12
        wave = 0.35 + 0.65 * abs(math.sin(orb.t * 9 + i * 0.9)) if open_ else 0.25
        hh = cr * (0.05 + 0.18 * open_ * wave)
        p.setBrush(_a(color.lighter(130), 0.9 if open_ else 0.55))
        p.drawRoundedRect(QRectF(x - cr * 0.035, my - hh / 2, cr * 0.07, hh), cr * 0.03, cr * 0.03)


# -- cat ----------------------------------------------------------------------------------------
def _cat(p, orb, cx, cy, cr, color, mood, blinking) -> None:
    fur = QColor("#f2b880")
    fur_dark = QColor("#c98a52")
    # ears (behind the head)
    for side in (-1, 1):
        ear = QPainterPath()
        ear.moveTo(cx + side * cr * 0.85, cy - cr * 0.25)
        ear.lineTo(cx + side * cr * 0.78, cy - cr * 1.12)
        ear.lineTo(cx + side * cr * 0.2, cy - cr * 0.78)
        ear.closeSubpath()
        p.setPen(QPen(_a(color, 0.8), cr * 0.04))
        p.setBrush(fur)
        p.drawPath(ear)
        inner = QPainterPath()
        inner.moveTo(cx + side * cr * 0.72, cy - cr * 0.42)
        inner.lineTo(cx + side * cr * 0.7, cy - cr * 0.94)
        inner.lineTo(cx + side * cr * 0.34, cy - cr * 0.74)
        inner.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#ffb3c6"))
        p.drawPath(inner)
    # head
    grad = QRadialGradient(QPointF(cx - cr * 0.3, cy - cr * 0.35), cr * 1.4)
    grad.setColorAt(0, fur.lighter(115))
    grad.setColorAt(1, fur_dark)
    p.setBrush(grad)
    p.setPen(QPen(_a(color, 0.8), cr * 0.045))
    p.drawEllipse(QPointF(cx, cy), cr * 0.95, cr * 0.88)
    # muzzle
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#fff1e0"))
    p.drawEllipse(QPointF(cx, cy + cr * 0.32), cr * 0.42, cr * 0.3)
    # eyes
    look = _look(orb, cr, 0.06, 0.05)
    eye_c = QColor("#6fd48b") if mood != "sad" else QColor("#8fb7d8")
    for side in (-1, 1):
        c = QPointF(cx + side * cr * 0.36, cy - cr * 0.12)
        if mood == "love":
            _heart_eye(p, c, cr * 0.22)
        elif mood == "sleep" or blinking:
            _closed_eye(p, c, cr * 0.28, QColor("#5a3a20"), cr * 0.05)
        elif mood == "happy":
            _closed_eye(p, c, cr * 0.28, QColor("#5a3a20"), cr * 0.05, happy=True)
        else:
            h = cr * (0.2 if mood != "think" else 0.14)
            p.setBrush(QColor("#2a1a0e"))
            p.drawEllipse(c, cr * 0.17, h + cr * 0.02)
            p.setBrush(eye_c)
            p.drawEllipse(c, cr * 0.14, h)
            pupil_w = cr * (0.035 + 0.04 * (1 if orb.state == "listening" else 0))
            p.setBrush(QColor("#140b05"))
            p.drawEllipse(QPointF(c.x() + look.x(), c.y() + look.y()), pupil_w, h * 0.85)
            p.setBrush(QColor(255, 255, 255, 220))
            p.drawEllipse(QPointF(c.x() - cr * 0.05, c.y() - h * 0.45), cr * 0.035, cr * 0.035)
    # nose and mouth
    nose = QPainterPath()
    nose.moveTo(cx - cr * 0.07, cy + cr * 0.17)
    nose.lineTo(cx + cr * 0.07, cy + cr * 0.17)
    nose.lineTo(cx, cy + cr * 0.25)
    nose.closeSubpath()
    p.setBrush(QColor("#ff7aa2"))
    p.drawPath(nose)
    open_ = _talk_open(orb)
    p.setPen(QPen(QColor("#5a3a20"), cr * 0.035, cap=Qt.PenCapStyle.RoundCap))
    p.setBrush(Qt.BrushStyle.NoBrush)
    if open_ > 0.05:
        p.setBrush(QColor("#7a2b3a"))
        p.drawEllipse(QPointF(cx, cy + cr * 0.36), cr * 0.09, cr * (0.03 + 0.09 * open_))
    else:
        w = QPainterPath()  # the little "w" mouth
        w.moveTo(cx - cr * 0.14, cy + cr * 0.3)
        w.quadTo(cx - cr * 0.07, cy + cr * 0.38, cx, cy + cr * 0.27)
        w.quadTo(cx + cr * 0.07, cy + cr * 0.38, cx + cr * 0.14, cy + cr * 0.3)
        p.drawPath(w)
    # whiskers
    p.setPen(QPen(_a(QColor("#5a3a20"), 0.7), cr * 0.02, cap=Qt.PenCapStyle.RoundCap))
    for side in (-1, 1):
        for k, dy in enumerate((-0.04, 0.04, 0.12)):
            sway = math.sin(orb.t * 2 + k) * cr * 0.015
            p.drawLine(QPointF(cx + side * cr * 0.4, cy + cr * (0.25 + dy)),
                       QPointF(cx + side * cr * 1.05, cy + cr * (0.2 + dy * 1.6) + sway))
    if mood in ("love", "happy"):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 120, 150, 90))
        for side in (-1, 1):
            p.drawEllipse(QPointF(cx + side * cr * 0.58, cy + cr * 0.18), cr * 0.13, cr * 0.08)


# -- anime --------------------------------------------------------------------------------------
def _anime(p, orb, cx, cy, cr, color, mood, blinking) -> None:
    skin = QColor("#ffe2cf")
    grad = QRadialGradient(QPointF(cx - cr * 0.25, cy - cr * 0.3), cr * 1.3)
    grad.setColorAt(0, skin.lighter(106))
    grad.setColorAt(1, QColor("#f4c7ad"))
    p.setBrush(grad)
    p.setPen(QPen(_a(color, 0.85), cr * 0.045))
    p.drawEllipse(QPointF(cx, cy), cr * 0.93, cr * 0.93)
    # fringe (hair) in the accent colour
    hair = QPainterPath()
    hair.moveTo(cx - cr * 0.93, cy - cr * 0.05)
    hair.cubicTo(cx - cr * 0.95, cy - cr * 1.05, cx + cr * 0.95, cy - cr * 1.05, cx + cr * 0.93, cy - cr * 0.05)
    for i, x in enumerate((0.7, 0.4, 0.1, -0.2, -0.5)):
        tip_y = cy - cr * (0.28 if i % 2 else 0.4)
        hair.lineTo(cx + cr * (x + 0.08), tip_y)
        hair.lineTo(cx + cr * (x - 0.15), cy - cr * 0.62)
    hair.lineTo(cx - cr * 0.93, cy - cr * 0.05)
    hg = QLinearGradient(QPointF(cx, cy - cr), QPointF(cx, cy - cr * 0.2))
    hg.setColorAt(0, color.darker(160))
    hg.setColorAt(1, color.darker(110))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(hg)
    p.drawPath(hair)
    # big eyes
    look = _look(orb, cr, 0.05, 0.04)
    for side in (-1, 1):
        c = QPointF(cx + side * cr * 0.35, cy + cr * 0.05)
        if mood == "love":
            _heart_eye(p, c, cr * 0.26)
            continue
        if mood == "sleep" or blinking:
            _closed_eye(p, c, cr * 0.32, QColor("#4a2a1a"), cr * 0.05)
            continue
        if mood == "happy":
            _closed_eye(p, c, cr * 0.32, QColor("#4a2a1a"), cr * 0.055, happy=True)
            continue
        h = cr * (0.27 if mood != "think" else 0.2)
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(c, cr * 0.17, h)
        iris = QRadialGradient(QPointF(c.x() + look.x(), c.y() + look.y() + h * 0.2), h)
        iris.setColorAt(0, color.lighter(150))
        iris.setColorAt(0.6, color)
        iris.setColorAt(1, color.darker(170))
        p.setBrush(iris)
        p.drawEllipse(QPointF(c.x() + look.x(), c.y() + look.y() + h * 0.08), cr * 0.14, h * 0.86)
        p.setBrush(QColor("#1a0f14"))
        p.drawEllipse(QPointF(c.x() + look.x(), c.y() + look.y() + h * 0.12), cr * 0.065, h * 0.45)
        p.setBrush(QColor(255, 255, 255, 235))  # sparkles
        p.drawEllipse(QPointF(c.x() - cr * 0.05 + look.x(), c.y() - h * 0.32 + look.y()), cr * 0.05, cr * 0.05)
        p.drawEllipse(QPointF(c.x() + cr * 0.06 + look.x(), c.y() + h * 0.25 + look.y()), cr * 0.025, cr * 0.025)
        p.setPen(QPen(QColor("#3a2018"), cr * 0.04, cap=Qt.PenCapStyle.RoundCap))
        p.drawArc(QRectF(c.x() - cr * 0.19, c.y() - h - cr * 0.03, cr * 0.38, h * 0.9), 30 * 16, 120 * 16)
        p.setPen(Qt.PenStyle.NoPen)
    p.setPen(Qt.PenStyle.NoPen)  # closed eyes leave a pen set; blush and tears are soft fills
    if mood == "sad":
        p.setBrush(QColor(120, 190, 255, 200))
        p.drawEllipse(QPointF(cx + cr * 0.48, cy + cr * 0.35), cr * 0.04, cr * 0.07)
    # blush
    p.setBrush(QColor(255, 120, 150, 110 if mood in ("love", "happy") else 60))
    for side in (-1, 1):
        p.drawEllipse(QPointF(cx + side * cr * 0.55, cy + cr * 0.36), cr * 0.13, cr * 0.07)
    # mouth
    open_ = _talk_open(orb)
    if open_ > 0.05:
        p.setBrush(QColor("#c4435e"))
        p.drawEllipse(QPointF(cx, cy + cr * 0.5), cr * 0.08, cr * (0.03 + 0.08 * open_))
    else:
        p.setPen(QPen(QColor("#a0405a"), cr * 0.035, cap=Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        smile = QPainterPath()
        bend = -0.03 if mood == "sad" else 0.06
        smile.moveTo(cx - cr * 0.09, cy + cr * 0.48)
        smile.quadTo(cx, cy + cr * (0.48 + bend), cx + cr * 0.09, cy + cr * 0.48)
        p.drawPath(smile)


# -- minimal ------------------------------------------------------------------------------------
def _minimal(p, orb, cx, cy, cr, color, mood, blinking) -> None:
    pulse = 0.5 + 0.5 * math.sin(orb.t * (1.2 if mood == "sleep" else 2.6))
    level = min(1.0, orb.level * 1.8)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for i, (w, a) in enumerate(((cr * 0.09, 0.9), (cr * 0.03, 0.45), (cr * 0.015, 0.25))):
        r = cr * (0.78 + i * 0.12 + 0.05 * level)
        p.setPen(QPen(_a(color, a * (0.45 if mood == "sleep" else 1.0)), w))
        p.drawEllipse(QPointF(cx, cy), r, r)
    if mood == "think":  # a sweeping arc while working
        p.setPen(QPen(_a(color.lighter(150), 0.95), cr * 0.09, cap=Qt.PenCapStyle.RoundCap))
        start = int((orb.t * 360) % 360 * 16)
        p.drawArc(QRectF(cx - cr * 0.78, cy - cr * 0.78, cr * 1.56, cr * 1.56), start, 70 * 16)
    core = QRadialGradient(QPointF(cx, cy), cr * 0.5)
    core.setColorAt(0, _a(color.lighter(160), 0.95))
    core.setColorAt(1, _a(color, 0.0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(core)
    size = cr * (0.28 + 0.1 * pulse + 0.25 * level)
    if mood == "love":
        p.setBrush(QColor("#ff4f8b"))
        p.drawPath(heart_path(QPointF(cx, cy), size * 1.4))
    else:
        p.drawEllipse(QPointF(cx, cy), size * 1.6, size * 1.6)
        p.setBrush(_a(QColor("#ffffff"), 0.85 if mood != "sleep" else 0.35))
        p.drawEllipse(QPointF(cx, cy), size * 0.35, size * 0.35)
