"""The cute side of Jarvis, and the instructor look for practice mode.

For a few seconds after a great moment (a thank-you, a joke, a perfect quiz) the AI core turns into a
glossy, bubbly anime face, then morphs back. Kinds: love, happy, wink, star.

The tutor persona (practice mode) adds a graduation cap with a swinging tassel, round glasses with a
glint, floating chalk letters and a pointer that taps when the tutor corrects you.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient

from jarvis.ui.aiface import heart_path

KINDS = ("love", "happy", "wink", "star")


def star_path(c: QPointF, r_out: float, r_in: float, points: int = 5, rot: float = -math.pi / 2) -> QPainterPath:
    path = QPainterPath()
    for k in range(points * 2):
        r = r_out if k % 2 == 0 else r_in
        a = rot + k * math.pi / points
        pt = QPointF(c.x() + math.cos(a) * r, c.y() + math.sin(a) * r)
        path.moveTo(pt) if k == 0 else path.lineTo(pt)
    path.closeSubpath()
    return path


def draw_cute(p: QPainter, orb, cx: float, cy: float, cr: float, color: QColor, kind: str) -> None:
    """A glossy bubble with a big expressive face."""
    t, level = orb.t, orb.level
    # body
    core = QRadialGradient(QPointF(cx - cr * 0.25, cy - cr * 0.3), cr * 1.3)
    core.setColorAt(0.0, QColor(color).lighter(175))
    mid = QColor(color)
    mid.setAlpha(225)
    core.setColorAt(0.5, mid)
    edge = QColor(color).darker(230)
    edge.setAlpha(240)
    core.setColorAt(1.0, edge)
    p.setPen(QPen(QColor(color).lighter(160), 2))
    p.setBrush(core)
    p.drawEllipse(QPointF(cx, cy), cr, cr)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 255, 255, 60))
    p.drawEllipse(QPointF(cx - cr * 0.35, cy - cr * 0.45), cr * 0.28, cr * 0.16)

    dark = QColor(color).darker(420)
    dark.setAlpha(240)
    ex, ey = cr * 0.34, cy - cr * 0.08
    ew, eh = cr * 0.17, cr * 0.30
    line = QPen(dark, max(2.0, cr * 0.06))
    line.setCapStyle(Qt.PenCapStyle.RoundCap)
    pink = QColor(255, 80, 140)
    bounce = 1 + 0.12 * max(0.0, math.sin(t * 9))

    def anime_eye(c: QPointF, scale: float = 1.0) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(dark)
        p.drawEllipse(c, ew * scale, eh * scale)
        p.setBrush(QColor(255, 255, 255, 235))
        p.drawEllipse(c + QPointF(-ew * 0.3, -eh * 0.4) * scale, ew * 0.4 * scale, ew * 0.4 * scale)
        p.setBrush(QColor(255, 255, 255, 160))
        p.drawEllipse(c + QPointF(ew * 0.35, eh * 0.35) * scale, ew * 0.2 * scale, ew * 0.2 * scale)

    def arc_eye(c: QPointF) -> None:
        path = QPainterPath()
        path.moveTo(c.x() - ew, c.y() + eh * 0.2)
        path.quadTo(c.x(), c.y() - eh * 0.75, c.x() + ew, c.y() + eh * 0.2)
        p.setPen(line)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    for side in (-1, 1):
        c = QPointF(cx + side * ex, ey)
        if kind == "love":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(pink)
            p.drawPath(heart_path(c, ew * 1.3 * bounce))
            p.setBrush(QColor(255, 255, 255, 170))
            p.drawEllipse(c + QPointF(-ew * 0.45, -ew * 0.35), ew * 0.22, ew * 0.22)
        elif kind == "star":
            spin = t * 1.5 * side
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 210, 60))
            p.drawPath(star_path(c, ew * 1.35 * bounce, ew * 0.6 * bounce, rot=-math.pi / 2 + spin))
            p.setBrush(QColor(255, 255, 255, 200))
            p.drawEllipse(c + QPointF(-ew * 0.25, -ew * 0.3), ew * 0.2, ew * 0.2)
        elif kind == "wink" and side == 1:
            arc_eye(c)
        elif kind == "happy":
            arc_eye(c)
        else:
            anime_eye(c, 1.08)

    # blush
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(255, 110, 150, 125))
    for side in (-1, 1):
        p.drawEllipse(QPointF(cx + side * ex * 1.45, ey + eh * 0.95), ew * 0.95, ew * 0.48)

    # mouth
    my = cy + cr * 0.36
    mw = cr * 0.18
    if kind in ("happy", "star"):  # open laughing mouth
        open_h = mw * (0.75 + 0.25 * math.sin(t * 8) + level * 0.4)
        mouth = QPainterPath()
        mouth.moveTo(cx - mw, my - mw * 0.15)
        mouth.quadTo(cx, my - mw * 0.35, cx + mw, my - mw * 0.15)
        mouth.quadTo(cx, my + open_h * 1.5, cx - mw, my - mw * 0.15)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(dark)
        p.drawPath(mouth)
        p.setBrush(QColor(255, 120, 140))
        p.drawEllipse(QPointF(cx, my + open_h * 0.55), mw * 0.45, open_h * 0.28)
    elif kind == "wink":  # smile with a little tongue
        path = QPainterPath()
        path.moveTo(cx - mw, my - mw * 0.2)
        path.quadTo(cx, my + mw * 0.9, cx + mw, my - mw * 0.2)
        p.setPen(line)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 110, 130))
        p.drawEllipse(QPointF(cx + mw * 0.35, my + mw * 0.45), mw * 0.28, mw * 0.32)
    else:  # sweet smile
        path = QPainterPath()
        path.moveTo(cx - mw, my - mw * 0.2)
        path.quadTo(cx, my + mw * 0.8, cx + mw, my - mw * 0.2)
        p.setPen(line)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)


def draw_sparkles(p: QPainter, sparkles: list, cx: float, cy: float, cr: float) -> None:
    """Little stars and hearts bursting out when the face changes."""
    p.setPen(Qt.PenStyle.NoPen)
    for age, angle, speed, kind in sparkles:
        if age < 0:
            continue
        d = cr * (1.0 + speed * age * 1.6)
        c = QPointF(cx + math.cos(angle) * d, cy + math.sin(angle) * d)
        alpha = int(255 * (1 - age) ** 1.5)
        size = cr * (0.09 + 0.05 * (1 - age))
        if kind == "heart":
            p.setBrush(QColor(255, 90, 150, alpha))
            p.drawPath(heart_path(c, size))
        else:
            p.setBrush(QColor(255, 215, 90, alpha))
            p.drawPath(star_path(c, size * 1.2, size * 0.5, 4, rot=age * 4))


# -- instructor look -------------------------------------------------------------------------
def draw_tutor(p: QPainter, orb, cx: float, cy: float, cr: float, color: QColor, cute: float) -> None:
    """Graduation cap, glasses (AI look only), pointer and chalk letters."""
    t = orb.t
    tilt = math.radians(-8 + 3 * math.sin(t * 1.3))
    # cap: board (a flat diamond seen in perspective) + band
    p.save()
    p.translate(cx, cy - cr * 0.98)
    p.rotate(math.degrees(tilt))
    band = QPainterPath()
    band.addRoundedRect(QRectF(-cr * 0.46, -cr * 0.02, cr * 0.92, cr * 0.26), cr * 0.06, cr * 0.06)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(22, 28, 46))
    p.drawPath(band)
    board = QPainterPath()
    board.moveTo(-cr * 0.95, 0)
    board.lineTo(0, -cr * 0.32)
    board.lineTo(cr * 0.95, 0)
    board.lineTo(0, cr * 0.28)
    board.closeSubpath()
    grad = QLinearGradient(QPointF(0, -cr * 0.32), QPointF(0, cr * 0.28))
    grad.setColorAt(0.0, QColor(52, 62, 92))
    grad.setColorAt(1.0, QColor(20, 26, 44))
    p.setBrush(grad)
    p.setPen(QPen(QColor(color).lighter(140), 1.5))
    p.drawPath(board)
    # button + tassel swinging
    gold = QColor(255, 200, 70)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(gold)
    p.drawEllipse(QPointF(0, -cr * 0.02), cr * 0.05, cr * 0.035)
    swing = 0.35 * math.sin(t * 2.2) + 0.5 * orb.level
    end = QPointF(cr * 0.62 + math.sin(swing) * cr * 0.1, cr * 0.05 + math.cos(swing) * cr * 0.45)
    cord = QPainterPath()
    cord.moveTo(0, -cr * 0.02)
    cord.quadTo(cr * 0.45, -cr * 0.05, cr * 0.62, cr * 0.06)
    cord.lineTo(end)
    pen = QPen(gold, max(1.5, cr * 0.025))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(cord)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(gold)
    p.drawRoundedRect(QRectF(end.x() - cr * 0.05, end.y(), cr * 0.1, cr * 0.16), cr * 0.03, cr * 0.03)
    p.restore()

    # glasses on the AI face (the cute face shows its own big eyes)
    if cute < 0.5 and getattr(orb, "face", True):
        ex, ey = cr * 0.33, cy - cr * 0.07
        rr = cr * 0.22
        frame = QPen(QColor(230, 240, 255, int(220 * (1 - cute * 2))), max(1.5, cr * 0.03))
        p.setPen(frame)
        p.setBrush(QColor(255, 255, 255, 14))
        for side in (-1, 1):
            p.drawEllipse(QPointF(cx + side * ex, ey), rr, rr * 0.86)
        p.drawLine(QPointF(cx - ex + rr, ey), QPointF(cx + ex - rr, ey))
        # glint sweeping across the lenses
        g = (t * 0.35) % 1.0
        if g < 0.35:
            k = g / 0.35
            p.setPen(QPen(QColor(255, 255, 255, int(180 * math.sin(math.pi * k))), max(1.0, cr * 0.025)))
            for side in (-1, 1):
                x = cx + side * ex - rr * 0.6 + k * rr * 1.2
                p.drawLine(QPointF(x, ey - rr * 0.5), QPointF(x + rr * 0.25, ey - rr * 0.1))

    # pointer: taps toward the screen when the tutor corrects something
    if orb.point_t > 0:
        k = orb.point_t
        tap = math.sin(t * 14) * 0.12 * k
        p.save()
        p.translate(cx + cr * 1.05, cy + cr * 0.35)
        p.rotate(-35 + math.degrees(tap))
        p.setOpacity(min(1.0, k * 2))
        stick = QPen(QColor(160, 110, 60), max(2.0, cr * 0.05))
        stick.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(stick)
        p.drawLine(QPointF(0, 0), QPointF(cr * 0.75, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255))
        p.drawEllipse(QPointF(cr * 0.78, 0), cr * 0.05, cr * 0.05)
        p.restore()
        p.setOpacity(1.0)

    # chalk letters drifting up
    if orb.letters:
        p.setFont(QFont("Segoe Print", max(8, int(cr * 0.16)), QFont.Weight.Bold))
        for age, drift, ch in orb.letters:
            c = QColor(240, 245, 250, int(200 * (1 - age)))
            p.setPen(c)
            p.drawText(QPointF(cx + cr * (drift * 1.6), cy - cr * (1.25 + age * 1.1)), ch)
