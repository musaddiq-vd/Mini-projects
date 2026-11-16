"""Small window widgets: app icon, CPU/RAM/battery meters, status dot and typing dots."""

from __future__ import annotations

import psutil
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QWidget

from jarvis.ui import theme
from jarvis.ui.theme import DANGER, WARN


def app_icon(color: str) -> QIcon:
    pix = QPixmap(64, 64)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QColor(color)
    p.setBrush(c.darker(200))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(4, 4, 56, 56)
    p.setBrush(c)
    p.drawEllipse(14, 14, 36, 36)
    p.setBrush(c.darker(400))
    p.drawEllipse(22, 26, 6, 10)  # tiny eyes
    p.drawEllipse(36, 26, 6, 10)
    p.end()
    return QIcon(pix)


class StatsBar(QWidget):
    """Tiny HUD meters: CPU, RAM and battery."""

    def __init__(self, accent: str, parent=None):
        super().__init__(parent)
        self.accent = accent
        self.values: list[tuple[str, float, str]] = []
        self.setFixedHeight(34)
        psutil.cpu_percent(interval=None)  # prime the counter
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(2000)
        self.refresh()

    def refresh(self) -> None:
        vals = [("CPU", psutil.cpu_percent(interval=None), ""), ("RAM", psutil.virtual_memory().percent, "")]
        try:
            bat = psutil.sensors_battery()
        except Exception:
            bat = None
        if bat:
            vals.append(("BAT", bat.percent, "⚡" if bat.power_plugged else ""))
        self.values = vals
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        n = len(self.values) or 1
        gap = 12
        cell = (self.width() - gap * (n - 1)) / n
        p.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        for i, (name, value, extra) in enumerate(self.values):
            x = i * (cell + gap)
            bad = value > 90 if name != "BAT" else value < 20 and not extra
            color = QColor(DANGER if bad else WARN if (name != "BAT" and value > 70) else theme.readable(self.accent))
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(x, 0, cell, 14), Qt.AlignmentFlag.AlignLeft, name)
            p.setPen(color)
            p.drawText(QRectF(x, 0, cell, 14), Qt.AlignmentFlag.AlignRight, f"{extra}{value:.0f}%")
            track = QRectF(x, 19, cell, 5)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.P.track)
            p.drawRoundedRect(track, 2.5, 2.5)
            p.setBrush(color)
            p.drawRoundedRect(QRectF(x, 19, cell * min(value, 100) / 100, 5), 2.5, 2.5)
        p.end()


class StatusDot(QWidget):
    """Pulsing dot next to the state label."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.color = QColor(theme.MUTED)
        self.t = 0.0
        self.setFixedSize(14, 14)
        timer = QTimer(self)
        timer.timeout.connect(self._tick)
        timer.start(50)

    def _tick(self) -> None:
        self.t += 0.05
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        import math

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        halo = QColor(self.color)
        halo.setAlpha(int(60 + 60 * math.sin(self.t * 4)))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(halo)
        p.drawEllipse(1, 1, 12, 12)
        p.setBrush(self.color)
        p.drawEllipse(4, 4, 6, 6)
        p.end()


class TypingDots(QWidget):
    """"Jarvis is thinking" with three bouncing dots."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.text = ""
        self.color = QColor(theme.MUTED)
        self.t = 0.0
        self.setFixedHeight(22)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def show_text(self, text: str, color: str) -> None:
        self.text, self.color = text, QColor(color)
        if not self._timer.isActive():
            self._timer.start(40)
        self.show()

    def hide_now(self) -> None:
        self._timer.stop()
        self.hide()

    def _tick(self) -> None:
        self.t += 0.04
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        import math

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        for k in range(3):
            c = QColor(self.color)
            bounce = max(0.0, math.sin(self.t * 6 - k * 0.7))
            c.setAlpha(int(110 + 145 * bounce))
            p.setBrush(c)
            p.drawEllipse(QRectF(8 + k * 12, 8 - 4 * bounce, 7, 7))
        p.setPen(QColor(theme.MUTED))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(QRectF(52, 0, self.width() - 52, self.height()), Qt.AlignmentFlag.AlignVCenter, self.text)
        p.end()
