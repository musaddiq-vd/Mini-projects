"""The animated orb: a living HUD with a cute face that reacts to everything Jarvis does.

- listening: open eyes that look around and blink, ripples when you talk
- speaking: happy eyes, a moving mouth, blush, a ring of spectrum bars
- thinking / working: eyes look up, satellites orbit, a dashed ring spins
- sleeping: eyes closed, slow breathing, floating "Z z z"
- connecting: radar sweep
- flash() on wake-up, pulse() after an action, action("text") floats a label
"""

from __future__ import annotations

import math
import random
import time
from collections import deque

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QConicalGradient, QFont, QPainter, QPainterPath, QPen, QRadialGradient
from PyQt6.QtWidgets import QWidget

from jarvis.ui.theme import DANGER, OK, state_color

N_BARS = 72


class Orb(QWidget):
    def __init__(self, accent: str, parent=None, face: bool = True):
        super().__init__(parent)
        self.accent = accent
        self.face = face
        self.state = "connecting"
        self.target = 0.0  # latest audio level 0..1
        self.level = 0.0  # smoothed
        self.phase = 0.0
        self.t = 0.0
        self._last = time.monotonic()
        self.history: deque[float] = deque([0.0] * 64, maxlen=64)
        self.bars = [0.0] * N_BARS
        self.stars = [self._new_star(random.random()) for _ in range(60)]
        self.ripples: list[list] = []  # [age, QColor]
        self.floaters: list[list] = []  # [text, age, QColor]
        self.zzz: list[list] = []  # [age, drift]
        self._zzz_timer = 0.0
        self.flash_t = 0.0
        self.blink_at = time.monotonic() + 2.5
        self.look = QPointF(0, 0)
        self.look_target = QPointF(0, 0)
        self._look_change = 0.0
        self._ripple_cool = 0.0
        self.setMinimumSize(280, 320)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    # -- public ----------------------------------------------------------------
    def set_level(self, value: float) -> None:
        self.target = max(self.target, value)

    def set_state(self, state: str) -> None:
        self.state = state

    def set_accent(self, accent: str) -> None:
        self.accent = accent

    def set_face(self, on: bool) -> None:
        self.face = on

    def flash(self) -> None:
        """Wake-up burst."""
        self.flash_t = 1.0
        self.ripples.append([0.0, QColor("#ffffff")])

    def pulse(self, ok: bool = True) -> None:
        self.ripples.append([0.0, QColor(OK if ok else DANGER)])

    def action(self, text: str, color: str | None = None) -> None:
        self.floaters.append([text, 0.0, QColor(color or self.accent)])
        self.floaters = self.floaters[-4:]

    # -- animation ---------------------------------------------------------------
    @staticmethod
    def _new_star(y: float | None = None) -> list[float]:
        # x, y (0..1), speed, size, twinkle phase
        return [random.random(), 1.05 if y is None else y, random.uniform(0.004, 0.02),
                random.uniform(0.6, 2.0), random.uniform(0, math.tau)]

    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(0.05, now - self._last)
        self._last = now
        self.t += dt
        self.level += (self.target - self.level) * 0.35
        self.target *= 0.82  # decay between audio chunks
        asleep = self.state in {"sleeping", "offline"}
        speed = 0.006 if asleep else 0.02 + self.level * 0.06
        if self.state == "thinking":
            speed = 0.05
        self.phase = (self.phase + speed) % (math.tau * 100)
        self.history.append(self.level)

        # spectrum bars: smooth random heights driven by the level
        active = self.state in {"speaking", "listening"}
        for k in range(N_BARS):
            wobble = 0.5 + 0.5 * math.sin(self.t * 7 + k * 0.9) * math.sin(self.t * 3.1 + k * 0.37)
            goal = self.level * (0.35 + 0.65 * wobble) if active else 0.0
            self.bars[k] += (goal - self.bars[k]) * 0.3

        # stars drift up slowly
        for s in self.stars:
            s[1] -= s[2] * dt * (0.4 if asleep else 1.0)
            s[4] += dt * 2
            if s[1] < -0.05:
                s[:] = self._new_star()

        # ripples when you talk, plus any queued ones
        self._ripple_cool -= dt
        if self.state == "listening" and self.level > 0.3 and self._ripple_cool <= 0:
            self.ripples.append([0.0, QColor(self.accent)])
            self._ripple_cool = 0.35
        for r in self.ripples:
            r[0] += dt / 1.3
        self.ripples = [r for r in self.ripples if r[0] < 1.0][-8:]

        for f in self.floaters:
            f[1] += dt / 2.2
        self.floaters = [f for f in self.floaters if f[1] < 1.0]

        if self.state == "sleeping":
            self._zzz_timer -= dt
            if self._zzz_timer <= 0:
                self.zzz.append([0.0, random.uniform(-0.3, 0.3)])
                self._zzz_timer = 1.1
        for z in self.zzz:
            z[0] += dt / 3.0
        self.zzz = [z for z in self.zzz if z[0] < 1.0]

        self.flash_t = max(0.0, self.flash_t - dt * 1.6)

        # eyes wander a little
        self._look_change -= dt
        if self._look_change <= 0:
            self._look_change = random.uniform(1.2, 3.5)
            if self.state == "thinking":
                self.look_target = QPointF(random.choice([-0.5, 0.5]), -0.6)
            elif self.state == "listening":
                self.look_target = QPointF(random.uniform(-0.35, 0.35), random.uniform(-0.2, 0.2))
            else:
                self.look_target = QPointF(0, 0)
        self.look += (self.look_target - self.look) * 0.08
        self.update()

    # -- painting ----------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        wave_h = 56
        cx, cy = w / 2, (h - wave_h) / 2
        base = min(w, h - wave_h) * 0.26
        color = state_color(self.state, self.accent)
        breathing = 0.05 * math.sin(self.t * 1.6) if self.state == "sleeping" else 0.015 * math.sin(self.t * 2.4)
        r = base * (1 + self.level * 0.22 + breathing + self.flash_t * 0.12)

        self._stars(p, w, h - wave_h, color)
        self._glow(p, cx, cy, r, color, w, h - wave_h)
        self._ripples(p, cx, cy, r)
        if self.state == "connecting":
            self._radar(p, cx, cy, r, color)
        self._arcs(p, cx, cy, r, color)
        self._spectrum(p, cx, cy, r, color)
        self._ticks(p, cx, cy, r, color)
        if self.state == "thinking":
            self._satellites(p, cx, cy, r, color)
        core_r = r * 0.62
        self._core(p, cx, cy, core_r, color)
        if self.face:
            self._face(p, cx, cy, core_r, color)
        self._zzzs(p, cx, cy, core_r)
        self._floaters(p, cx, min(cy + r * 2.0, h - wave_h - 14))
        self._wave(p, w, h, wave_h, color)
        p.end()

    def _stars(self, p, w, h, color) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        for x, y, _, size, tw in self.stars:
            c = QColor(color)
            c.setAlpha(int(25 + 70 * (0.5 + 0.5 * math.sin(tw))))
            p.setBrush(c)
            p.drawEllipse(QPointF(x * w, y * h), size, size)

    def _glow(self, p, cx, cy, r, color, w, h) -> None:
        glow_r = min(r * 2.1, w / 2, h / 2)  # stay inside the widget so no square edges
        glow = QRadialGradient(QPointF(cx, cy), glow_r)
        g = QColor(color)
        g.setAlpha(min(255, int(60 + 120 * self.level + 120 * self.flash_t)))
        glow.setColorAt(0.0, g)
        g.setAlpha(0)
        glow.setColorAt(1.0, g)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(glow)
        p.drawEllipse(QPointF(cx, cy), glow_r, glow_r)

    def _ripples(self, p, cx, cy, r) -> None:
        p.setBrush(Qt.BrushStyle.NoBrush)
        for age, col in self.ripples:
            c = QColor(col)
            c.setAlpha(int(200 * (1 - age) ** 2))
            p.setPen(QPen(c, 2.5 * (1 - age) + 0.5))
            rr = r * (0.7 + 1.6 * age)
            p.drawEllipse(QPointF(cx, cy), rr, rr)

    def _radar(self, p, cx, cy, r, color) -> None:
        grad = QConicalGradient(QPointF(cx, cy), -math.degrees(self.t * 2.5))
        c = QColor(color)
        c.setAlpha(110)
        grad.setColorAt(0.0, c)
        c.setAlpha(0)
        grad.setColorAt(0.25, c)
        grad.setColorAt(1.0, c)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawEllipse(QPointF(cx, cy), r * 1.7, r * 1.7)

    def _arcs(self, p, cx, cy, r, color) -> None:
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i, (scale, span, width, direction) in enumerate(
            [(1.45, 70, 2.0, 1), (1.30, 110, 3.0, -1), (1.62, 40, 1.5, 1), (1.18, 200, 1.0, -1)]
        ):
            c = QColor(color)
            c.setAlpha(110 + int(100 * self.level) if i < 2 else 80)
            pen = QPen(c, width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            rr = r * scale
            start = int(math.degrees(self.phase * (1.4 + i * 0.5) * direction) * 16)
            for k in range(2):
                p.drawArc(QRectF(cx - rr, cy - rr, rr * 2, rr * 2), start + k * 180 * 16, span * 16)
        # dashed outer ring, spins fast while working
        c = QColor(color)
        c.setAlpha(90 if self.state == "thinking" else 45)
        pen = QPen(c, 1.5)
        pen.setDashPattern([3, 6])
        pen.setDashOffset(-self.phase * (40 if self.state == "thinking" else 8))
        p.setPen(pen)
        p.drawEllipse(QPointF(cx, cy), r * 1.95, r * 1.95)

    def _spectrum(self, p, cx, cy, r, color) -> None:
        if max(self.bars) < 0.01:
            return
        c = QColor(color).lighter(130)
        for k, v in enumerate(self.bars):
            if v < 0.01:
                continue
            ang = math.tau * k / N_BARS - math.pi / 2
            r1 = r * 0.68
            r2 = r1 + r * 0.55 * v
            c.setAlpha(int(90 + 160 * v))
            pen = QPen(c, 2.2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(QPointF(cx + math.cos(ang) * r1, cy + math.sin(ang) * r1),
                       QPointF(cx + math.cos(ang) * r2, cy + math.sin(ang) * r2))

    def _ticks(self, p, cx, cy, r, color) -> None:
        c = QColor(color)
        c.setAlpha(60)
        p.setPen(QPen(c, 1))
        for k in range(60):
            ang = math.tau * k / 60 + self.phase * 0.3
            r1, r2 = r * 1.75, r * (1.80 if k % 5 else 1.88)
            p.drawLine(QPointF(cx + math.cos(ang) * r1, cy + math.sin(ang) * r1),
                       QPointF(cx + math.cos(ang) * r2, cy + math.sin(ang) * r2))

    def _satellites(self, p, cx, cy, r, color) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        for k in range(3):
            ang = self.t * 3.2 + k * math.tau / 3
            for trail in range(6):
                a = ang - trail * 0.12
                c = QColor(color).lighter(140)
                c.setAlpha(int(230 * (1 - trail / 6)))
                p.setBrush(c)
                rr = r * 1.38
                p.drawEllipse(QPointF(cx + math.cos(a) * rr, cy + math.sin(a) * rr),
                              4 - trail * 0.5, 4 - trail * 0.5)

    def _core(self, p, cx, cy, cr, color) -> None:
        core = QRadialGradient(QPointF(cx - cr * 0.25, cy - cr * 0.3), cr * 1.3)
        c1 = QColor(color).lighter(170 + int(80 * self.flash_t))
        c2 = QColor(color)
        c2.setAlpha(210)
        c3 = QColor(color).darker(260)
        c3.setAlpha(235)
        core.setColorAt(0.0, c1)
        core.setColorAt(0.5, c2)
        core.setColorAt(1.0, c3)
        p.setPen(QPen(QColor(color).lighter(150), 2))
        p.setBrush(core)
        p.drawEllipse(QPointF(cx, cy), cr, cr)
        # glossy highlight
        hl = QColor(255, 255, 255, 50)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(hl)
        p.drawEllipse(QPointF(cx - cr * 0.35, cy - cr * 0.45), cr * 0.28, cr * 0.16)

    def _face(self, p, cx, cy, cr, color) -> None:
        dark = QColor(color).darker(400)
        dark.setAlpha(235)
        state = self.state
        ex = cr * 0.34
        ey = cy - cr * 0.08
        ew, eh = cr * 0.17, cr * 0.30
        look = QPointF(self.look.x() * cr * 0.08, self.look.y() * cr * 0.08)

        now = time.monotonic()
        if now > self.blink_at + 0.14:
            self.blink_at = now + random.uniform(2.0, 5.5)
        blinking = self.blink_at <= now <= self.blink_at + 0.14

        line = QPen(dark, max(2.0, cr * 0.06))
        line.setCapStyle(Qt.PenCapStyle.RoundCap)

        for side in (-1, 1):
            center = QPointF(cx + side * ex, ey) + look
            if state in {"sleeping", "offline", "connecting"} or blinking:
                # closed eyes: gentle downward curves
                path = QPainterPath()
                path.moveTo(center.x() - ew, center.y())
                path.quadTo(center.x(), center.y() + eh * 0.55, center.x() + ew, center.y())
                p.setPen(line)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(path)
            elif state == "speaking" and self.level > 0.25:
                # happy ^ ^ eyes
                path = QPainterPath()
                path.moveTo(center.x() - ew, center.y() + eh * 0.2)
                path.quadTo(center.x(), center.y() - eh * 0.7, center.x() + ew, center.y() + eh * 0.2)
                p.setPen(line)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(path)
            else:
                # big anime eyes with sparkles
                h = eh * (1.12 if self.flash_t > 0.2 else 1.0)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(dark)
                p.drawEllipse(center, ew, h)
                p.setBrush(QColor(255, 255, 255, 230))
                p.drawEllipse(center + QPointF(-ew * 0.3, -h * 0.4), ew * 0.38, ew * 0.38)
                p.setBrush(QColor(255, 255, 255, 150))
                p.drawEllipse(center + QPointF(ew * 0.35, h * 0.35), ew * 0.18, ew * 0.18)

        # blush while talking or just woken
        if state == "speaking" or self.flash_t > 0:
            blush = QColor(255, 110, 150, int(70 + 60 * max(self.level, self.flash_t)))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(blush)
            for side in (-1, 1):
                p.drawEllipse(QPointF(cx + side * ex * 1.45, ey + eh * 0.95), ew * 0.9, ew * 0.45)

        # mouth
        my = cy + cr * 0.36
        mw = cr * 0.16
        p.setPen(line)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if state == "speaking":
            open_h = cr * (0.04 + 0.22 * self.level)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(dark)
            p.drawEllipse(QPointF(cx, my), mw * (0.8 + self.level * 0.4), open_h)
        elif state == "thinking":
            p.drawEllipse(QPointF(cx + cr * 0.05, my), mw * 0.35, mw * 0.35)
        elif state == "muted":
            p.drawLine(QPointF(cx - mw, my), QPointF(cx + mw, my))
        elif state in {"sleeping", "offline"}:
            p.drawEllipse(QPointF(cx, my), mw * 0.3, mw * 0.22)
        else:
            path = QPainterPath()
            path.moveTo(cx - mw, my - mw * 0.2)
            path.quadTo(cx, my + mw * 0.7, cx + mw, my - mw * 0.2)
            p.drawPath(path)

    def _zzzs(self, p, cx, cy, cr) -> None:
        if not self.zzz:
            return
        for age, drift in self.zzz:
            size = int(cr * (0.18 + 0.25 * age))
            f = QFont("Segoe UI", max(8, size // 2), QFont.Weight.Bold)
            p.setFont(f)
            c = QColor("#cfe7ff")
            c.setAlpha(int(230 * (1 - age)))
            p.setPen(c)
            x = cx + cr * (0.7 + drift + age * 0.5)
            y = cy - cr * (0.7 + age * 1.4)
            p.drawText(QPointF(x, y), "z" if age < 0.35 else "Z")

    def _floaters(self, p, cx, bottom) -> None:
        """Action labels rise from below the orb and fade out."""
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        fm = p.fontMetrics()
        for i, (text, age, col) in enumerate(self.floaters):
            c = QColor(col)
            c.setAlpha(int(255 * min(1.0, (1 - age) * 2.5) * min(1.0, age * 8)))
            y = bottom - age * 34 - i * 2
            tw = fm.horizontalAdvance(text)
            rect = QRectF(cx - tw / 2 - 10, y - 14, tw + 20, 22)
            bg = QColor(10, 15, 20, int(c.alpha() * 0.75))
            p.setPen(QPen(c, 1))
            p.setBrush(bg)
            p.drawRoundedRect(rect, 11, 11)
            text_c = QColor(c).lighter(150)
            text_c.setAlpha(c.alpha())
            p.setPen(text_c)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _wave(self, p, w, h, wave_h, color) -> None:
        bars = list(self.history)
        n = len(bars)
        bw = (w - 40) / n
        mid = h - wave_h / 2
        c = QColor(color)
        p.setPen(Qt.PenStyle.NoPen)
        for k, v in enumerate(bars):
            amp = max(2.0, v * wave_h * 0.9)
            # brightest in the middle, fading to the edges
            fade = 1 - abs(k - n / 2) / (n / 2)
            c.setAlpha(50 + int(200 * fade))
            p.setBrush(c)
            x = 20 + k * bw
            p.drawRoundedRect(QRectF(x, mid - amp / 2, max(2.0, bw * 0.55), amp), 1.5, 1.5)
