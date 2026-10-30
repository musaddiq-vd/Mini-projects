"""The animated orb and waveform that react to the microphone and to Jarvis's voice."""

from __future__ import annotations

import math
from collections import deque

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen, QRadialGradient
from PyQt6.QtWidgets import QWidget

from jarvis.ui.theme import state_color


class Orb(QWidget):
    def __init__(self, accent: str, parent=None):
        super().__init__(parent)
        self.accent = accent
        self.state = "connecting"
        self.target = 0.0  # latest audio level 0..1
        self.level = 0.0  # smoothed
        self.phase = 0.0
        self.history: deque[float] = deque([0.0] * 56, maxlen=56)
        self.setMinimumSize(260, 300)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    def set_level(self, value: float) -> None:
        self.target = max(self.target, value)

    def set_state(self, state: str) -> None:
        self.state = state

    def set_accent(self, accent: str) -> None:
        self.accent = accent

    def _tick(self) -> None:
        self.level += (self.target - self.level) * 0.35
        self.target *= 0.82  # decay between audio chunks
        speed = 0.006 if self.state in {"sleeping", "offline"} else 0.02 + self.level * 0.06
        self.phase = (self.phase + speed) % (math.tau * 100)
        self.history.append(self.level)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        wave_h = 56
        cx, cy = w / 2, (h - wave_h) / 2
        base = min(w, h - wave_h) * 0.30
        color = state_color(self.state, self.accent)
        breathing = 0.04 * math.sin(self.phase * 2) if self.state == "sleeping" else 0
        r = base * (1 + self.level * 0.22 + breathing)

        # soft glow
        glow_r = min(r * 1.9, w / 2, (h - wave_h) / 2)  # stay inside the widget so no square edges
        glow = QRadialGradient(QPointF(cx, cy), glow_r)
        g = QColor(color)
        g.setAlpha(int(70 + 120 * self.level))
        glow.setColorAt(0.0, g)
        g.setAlpha(0)
        glow.setColorAt(1.0, g)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(glow)
        p.drawEllipse(QPointF(cx, cy), glow_r, glow_r)

        # rotating arcs
        for i, (scale, span, width, direction) in enumerate(
            [(1.45, 70, 2.0, 1), (1.30, 110, 3.0, -1), (1.62, 40, 1.5, 1), (1.18, 200, 1.0, -1)]
        ):
            c = QColor(color)
            c.setAlpha(110 + int(100 * self.level) if i < 2 else 80)
            pen = QPen(c, width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            rr = r * scale
            start = int(math.degrees(self.phase * (1.4 + i * 0.5) * direction) * 16)
            for k in range(2):
                p.drawArc(QRectF(cx - rr, cy - rr, rr * 2, rr * 2), start + k * 180 * 16, span * 16)

        # tick ring
        c = QColor(color)
        c.setAlpha(60)
        p.setPen(QPen(c, 1))
        for k in range(60):
            ang = math.tau * k / 60 + self.phase * 0.3
            r1, r2 = r * 1.75, r * (1.80 if k % 5 else 1.88)
            p.drawLine(QPointF(cx + math.cos(ang) * r1, cy + math.sin(ang) * r1),
                       QPointF(cx + math.cos(ang) * r2, cy + math.sin(ang) * r2))

        # core
        core = QRadialGradient(QPointF(cx, cy), r)
        c1, c2 = QColor(color).lighter(160), QColor(color)
        c2.setAlpha(200)
        core.setColorAt(0.0, c1)
        core.setColorAt(0.55, c2)
        c3 = QColor(color).darker(250)
        c3.setAlpha(230)
        core.setColorAt(1.0, c3)
        p.setPen(QPen(QColor(color).lighter(140), 2))
        p.setBrush(core)
        p.drawEllipse(QPointF(cx, cy), r * 0.62, r * 0.62)

        # waveform bars
        bars = list(self.history)
        n = len(bars)
        bw = (w - 40) / n
        top = h - wave_h / 2
        c = QColor(color)
        for k, v in enumerate(bars):
            amp = max(2.0, v * wave_h * 0.9)
            c.setAlpha(60 + int(180 * (k / n)))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            x = 20 + k * bw
            p.drawRoundedRect(QRectF(x, top - amp / 2, max(2.0, bw * 0.55), amp), 1.5, 1.5)
        p.end()
