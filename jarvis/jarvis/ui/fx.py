"""Visual polish shared by the windows: animated background, hover glow, toasts, typewriter text."""

from __future__ import annotations

import math
import random
import time

from PyQt6.QtCore import (
    QEasingCurve, QEvent, QObject, QParallelAnimationGroup, QPoint, QPointF, QPropertyAnimation, QRectF, Qt,
    QTimer, pyqtProperty,
)
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPen, QRadialGradient
from PyQt6.QtWidgets import QFrame, QGraphicsDropShadowEffect, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QWidget


class AnimatedBackground(QWidget):
    """Slowly drifting aurora blobs, a faint perspective grid and dust. Repaints at ~30 fps."""

    def __init__(self, accent: str, parent=None):
        super().__init__(parent)
        self.accent = QColor(accent)
        self.t = random.uniform(0, 100)
        self.dust = [[random.random(), random.random(), random.uniform(0.002, 0.008), random.uniform(0.5, 1.6)]
                     for _ in range(45)]
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def set_accent(self, accent: str) -> None:
        self.accent = QColor(accent)

    def _tick(self) -> None:
        if not self.isVisible():
            return
        self.t += 0.033
        for d in self.dust:
            d[1] -= d[2] * 0.033
            if d[1] < -0.02:
                d[0], d[1] = random.random(), 1.02
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        from jarvis.ui import theme

        base = QLinearGradient(0, 0, w, h)
        base.setColorAt(0.0, QColor(theme.P.bg_top))
        base.setColorAt(1.0, QColor(theme.P.bg_bottom))
        light = theme.P.light
        p.fillRect(self.rect(), base)

        # aurora blobs
        a = self.accent
        blobs = [
            (0.22 + 0.08 * math.sin(self.t * 0.13), 0.35 + 0.1 * math.cos(self.t * 0.11), 0.55, a, 30 if light else 38),
            (0.80 + 0.06 * math.cos(self.t * 0.09), 0.70 + 0.08 * math.sin(self.t * 0.12), 0.5,
             QColor(theme.SECOND), 24 if light else 26),
            (0.55 + 0.1 * math.sin(self.t * 0.07 + 2), 0.05 + 0.05 * math.cos(self.t * 0.1), 0.45, a, 18 if light else 20),
        ]
        p.setPen(Qt.PenStyle.NoPen)
        for bx, by, size, color, alpha in blobs:
            r = max(w, h) * size
            g = QRadialGradient(QPointF(bx * w, by * h), r)
            c = QColor(color)
            c.setAlpha(alpha)
            g.setColorAt(0.0, c)
            c.setAlpha(0)
            g.setColorAt(1.0, c)
            p.setBrush(g)
            p.drawEllipse(QPointF(bx * w, by * h), r, r)

        # faint grid, scrolling slowly
        grid = QColor(a)
        grid.setAlpha(16 if light else 10)
        p.setPen(QPen(grid, 1))
        step = 44
        off = (self.t * 6) % step
        for x in range(-step, w + step, step):
            p.drawLine(QPointF(x + off, 0), QPointF(x + off, h))
        for y in range(-step, h + step, step):
            p.drawLine(QPointF(0, y + off), QPointF(w, y + off))

        # dust
        p.setPen(Qt.PenStyle.NoPen)
        for x, y, _, size in self.dust:
            c = QColor(a)
            c.setAlpha(int(40 + 40 * math.sin(self.t * 2 + x * 20)))
            p.setBrush(c)
            p.drawEllipse(QPointF(x * w, y * h), size, size)
        p.end()


class _Glow(QObject):
    """Animates a drop-shadow glow on hover / focus."""

    def __init__(self, widget: QWidget, color: str, radius: float = 22):
        super().__init__(widget)
        self.effect = QGraphicsDropShadowEffect(widget)
        self.effect.setOffset(0, 0)
        self.effect.setBlurRadius(0)
        c = QColor(color)
        c.setAlpha(170)
        self.effect.setColor(c)
        widget.setGraphicsEffect(self.effect)
        self.radius = radius
        self.anim = QPropertyAnimation(self.effect, b"blurRadius", self)
        self.anim.setDuration(220)
        self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        widget.installEventFilter(self)

    def set_color(self, color: str) -> None:
        c = QColor(color)
        c.setAlpha(170)
        self.effect.setColor(c)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() in (QEvent.Type.Enter, QEvent.Type.FocusIn):
            self._go(self.radius)
        elif event.type() in (QEvent.Type.Leave, QEvent.Type.FocusOut):
            if not (obj.hasFocus() and event.type() == QEvent.Type.Leave):
                self._go(0)
        return False

    def _go(self, value: float) -> None:
        self.anim.stop()
        self.anim.setStartValue(self.effect.blurRadius())
        self.anim.setEndValue(value)
        self.anim.start()


def add_glow(widget: QWidget, color: str, radius: float = 22) -> _Glow:
    return _Glow(widget, color, radius)


class Toast(QFrame):
    """A small card that slides in at the top-right of its parent and fades away."""

    def __init__(self, parent: QWidget, icon: str, text: str, color: str, seconds: float = 4.5):
        super().__init__(parent)
        self.setObjectName("toast")
        from jarvis.ui import theme

        self.setStyleSheet(
            f"QFrame#toast {{ background: {theme.P.toast_bg}; border: 1px solid {color}; border-radius: 12px; }}"
            "QFrame#toast QLabel { background: transparent; }")
        h = QHBoxLayout(self)
        h.setContentsMargins(14, 10, 16, 10)
        h.setSpacing(10)
        ic = QLabel(icon)
        ic.setStyleSheet("font-size: 18px;")
        h.addWidget(ic)
        label = QLabel(text)
        label.setTextFormat(Qt.TextFormat.PlainText)  # titles can come from other people (calendar invites)
        label.setWordWrap(True)
        label.setMaximumWidth(300)
        label.setStyleSheet(f"color:{theme.TEXT}; font-size: 13px;")
        h.addWidget(label, 1)
        self.adjustSize()
        self.setFixedWidth(min(360, max(220, self.sizeHint().width())))
        self.adjustSize()
        self.opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.seconds = seconds

    def show_at(self, target: QPoint) -> None:
        start = QPoint(target.x() + 60, target.y())
        self.move(start)
        self.show()
        self.raise_()
        group = QParallelAnimationGroup(self)
        slide = QPropertyAnimation(self, b"pos", self)
        slide.setDuration(380)
        slide.setStartValue(start)
        slide.setEndValue(target)
        slide.setEasingCurve(QEasingCurve.Type.OutBack)
        fade = QPropertyAnimation(self.opacity, b"opacity", self)
        fade.setDuration(300)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        group.addAnimation(slide)
        group.addAnimation(fade)
        group.start()
        self._in = group
        QTimer.singleShot(int(self.seconds * 1000), self.dismiss)

    def dismiss(self) -> None:
        out = QPropertyAnimation(self.opacity, b"opacity", self)
        out.setDuration(400)
        out.setStartValue(self.opacity.opacity())
        out.setEndValue(0.0)
        out.finished.connect(self._gone_now)
        out.start()
        self._out = out

    gone = False

    def _gone_now(self) -> None:
        self.gone = True
        self.hide()
        if getattr(self, "hidden_cb", None):
            self.hidden_cb()
        self.deleteLater()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.dismiss()


class ToastHost:
    """Stacks toasts in the top-right corner of a widget."""

    def __init__(self, parent: QWidget, margin: int = 22, top: int = 64):
        self.parent = parent
        self.margin = margin
        self.top = top  # below the panel headers
        self.toasts: list[Toast] = []

    def show(self, icon: str, text: str, color: str, seconds: float = 4.5) -> Toast:
        self.toasts = [t for t in self.toasts if _alive(t)]
        toast = Toast(self.parent, icon, text, color, seconds)
        y = self.top + sum(t.height() + 8 for t in self.toasts)
        x = self.parent.width() - toast.width() - self.margin
        toast.destroyed.connect(lambda *_: self._restack())
        toast.hidden_cb = self._restack
        self.toasts.append(toast)
        toast.show_at(QPoint(x, y))
        return toast

    def _restack(self) -> None:
        try:
            self.parent.width()
        except (RuntimeError, AttributeError):  # window closing / app shutting down: nothing to arrange
            self.toasts = []
            return
        self.toasts = [t for t in self.toasts if _alive(t)]
        y = self.top
        for t in self.toasts:
            anim = QPropertyAnimation(t, b"pos", t)
            anim.setDuration(250)
            anim.setEndValue(QPoint(self.parent.width() - t.width() - self.margin, y))
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            anim.start()
            t._restack_anim = anim
            y += t.height() + 8


def _alive(widget) -> bool:
    try:
        widget.isVisible()
    except RuntimeError:
        return False
    return not getattr(widget, "gone", False)


class Typewriter(QObject):
    """Reveals a label's text letter by letter (quick: never longer than ~1 s)."""

    def __init__(self, label: QLabel, text: str, parent=None, final_html: str | None = None):
        super().__init__(parent or label)
        self.label, self.text, self.final_html = label, text, final_html
        label.setTextFormat(Qt.TextFormat.PlainText)
        self.pos = 0
        self.duration = min(0.9, 0.25 + len(text) * 0.008)  # seconds, by the clock (not by frames)
        self.started = time.monotonic()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    def _tick(self) -> None:
        if not _alive(self.label):
            self._timer.stop()
            return
        done = (time.monotonic() - self.started) / self.duration
        self.pos = min(len(self.text), max(self.pos + 1, int(len(self.text) * done)))
        cursor = "▍" if self.pos < len(self.text) else ""
        self.label.setText(self.text[: self.pos] + cursor)
        if self.pos >= len(self.text):
            self.finish()

    def finish(self) -> None:
        self._timer.stop()
        if not _alive(self.label):
            return
        if self.final_html is not None:  # formatting (bold, links...) appears once typing is done
            self.label.setTextFormat(Qt.TextFormat.RichText)
            self.label.setText(self.final_html)
        else:
            self.label.setText(self.text)


class _HeightAnimator(QObject):
    """Grows a widget from 0 to its natural height (used for new chat rows)."""

    def __init__(self, widget: QWidget, duration: int = 280):
        super().__init__(widget)
        self.widget = widget
        self._h = 0
        target = max(1, widget.sizeHint().height())
        self.anim = QPropertyAnimation(self, b"h", self)
        self.anim.setDuration(duration)
        self.anim.setStartValue(0)
        self.anim.setEndValue(target)
        self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.anim.finished.connect(lambda: widget.setMaximumHeight(16_777_215))
        widget.setMaximumHeight(0)
        self.anim.start()

    def _get(self) -> int:
        return self._h

    def _set(self, value: int) -> None:
        self._h = value
        if _alive(self.widget):
            self.widget.setMaximumHeight(value)

    h = pyqtProperty(int, _get, _set)


def grow_in(widget: QWidget, duration: int = 280) -> _HeightAnimator:
    return _HeightAnimator(widget, duration)


__all__ = ["AnimatedBackground", "ToastHost", "Typewriter", "add_glow", "grow_in", "QRectF"]


class Confetti(QWidget):
    """A burst of confetti over a widget for celebrations; removes itself when done."""

    COLORS = ("#ff5c8a", "#ffd23f", "#3ddc97", "#00d4ff", "#8b7bff", "#ff8a3d", "#ffffff")

    def __init__(self, parent: QWidget, pieces: int = 120, seconds: float = 2.8):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(parent.rect())
        w, h = max(1, self.width()), max(1, self.height())
        self.parts = []
        for _ in range(pieces):
            # x, y, vx, vy, angle, spin, size, color, shape
            self.parts.append([random.uniform(0.2, 0.8) * w, h * random.uniform(0.25, 0.45),
                               random.uniform(-380, 380), random.uniform(-760, -260), random.uniform(0, 360),
                               random.uniform(-540, 540), random.uniform(5, 10), QColor(random.choice(self.COLORS)),
                               random.choice(("rect", "circle", "strip"))])
        self.age, self.life = 0.0, seconds
        self._last = time.monotonic()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)
        self.show()
        self.raise_()

    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(0.05, now - self._last)
        self._last = now
        self.age += dt
        for p in self.parts:
            p[3] += 900 * dt  # gravity
            p[2] *= 0.985  # air
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[4] += p[5] * dt
        if self.age >= self.life:
            self._timer.stop()
            self.deleteLater()
            return
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        fade = max(0.0, min(1.0, (self.life - self.age) / 0.6))
        p.setPen(Qt.PenStyle.NoPen)
        for x, y, _, _, angle, _, size, color, shape in self.parts:
            c = QColor(color)
            c.setAlpha(int(235 * fade))
            p.save()
            p.translate(x, y)
            p.rotate(angle)
            p.setBrush(c)
            if shape == "circle":
                p.drawEllipse(QPointF(0, 0), size * 0.45, size * 0.45)
            elif shape == "strip":
                p.drawRoundedRect(QRectF(-size * 0.15, -size, size * 0.3, size * 2), 1, 1)
            else:
                p.drawRect(QRectF(-size / 2, -size / 3, size, size * 0.66))
            p.restore()
        p.end()
