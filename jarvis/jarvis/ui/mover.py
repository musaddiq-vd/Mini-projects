"""Moving Jarvis' own window by voice: nudge, place in a corner, glide until "stop", resize.

Works on whichever Jarvis window is showing (the full window or the mini orb) and always keeps it on screen.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from PyQt6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QRect, QSize, QTimer
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QWidget

STEP = {"little": 70, "some": 180, "lot": 420}
GLIDE_SPEED = {"slow": 180, "normal": 360, "fast": 720}  # pixels per second
GLIDE_MAX_S = 25
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
PLACES = ("top_left", "top_right", "bottom_left", "bottom_right", "center", "left_half", "right_half", "top",
          "bottom", "left", "right")
MARGIN = 12


class WindowMover:
    def __init__(self, target: Callable[[], QWidget]):
        self.target = target  # returns the window to move right now
        self._anim: QPropertyAnimation | None = None
        self._glide = QTimer()
        self._glide.setInterval(16)
        self._glide.timeout.connect(self._glide_step)
        self._dir = (0, 0)
        self._speed = 6
        self._glide_until = 0.0

    # -- helpers -----------------------------------------------------------------------------
    def _screen(self, w: QWidget) -> QRect:
        screen = w.screen() or QGuiApplication.primaryScreen()
        return screen.availableGeometry()

    def _ready(self) -> QWidget:
        w = self.target()
        if w.isMaximized() or w.isFullScreen():
            w.showNormal()  # a maximised window can't move
        return w

    def clamp(self, w: QWidget, pos: QPoint) -> QPoint:
        area, size = self._screen(w), w.frameGeometry().size()
        # a window bigger than the screen sticks to the top-left instead of sliding off it
        x = max(min(pos.x(), area.right() - size.width() + 1), area.left())
        y = max(min(pos.y(), area.bottom() - size.height() + 1), area.top())
        return QPoint(x, y)

    def _animate_to(self, w: QWidget, pos: QPoint, ms: int = 320) -> None:
        self.stop()
        anim = QPropertyAnimation(w, b"pos", w)
        anim.setDuration(ms)
        anim.setEndValue(self.clamp(w, pos))
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start()
        self._anim = anim

    # -- actions -----------------------------------------------------------------------------
    def nudge(self, direction: str, amount: str = "little") -> str:
        if direction not in DIRS:
            return "unknown direction"
        w = self._ready()
        dx, dy = DIRS[direction]
        step = STEP.get(amount, STEP["little"])
        target = self.clamp(w, w.pos() + QPoint(dx * step, dy * step))
        edge = target == self.clamp(w, w.pos()) and target == w.pos()
        self._animate_to(w, target)
        return "edge" if edge else "moved"

    def place(self, where: str) -> str:
        if where not in PLACES:
            return "unknown place"
        w = self._ready()
        area = self._screen(w)
        if where in ("left_half", "right_half"):
            self.stop()
            half = QRect(area.left(), area.top(), area.width() // 2, area.height())
            if where == "right_half":
                half.moveLeft(area.left() + area.width() // 2)
            frame, inner = w.frameGeometry(), w.geometry()  # title bar and borders sit outside geometry()
            client = half.adjusted(inner.left() - frame.left(), inner.top() - frame.top(),
                                   inner.right() - frame.right(), inner.bottom() - frame.bottom())
            if w.minimumWidth() <= client.width() <= w.maximumWidth():
                w.setGeometry(client)
                return "placed"
            where = "left" if where == "left_half" else "right"
        size = w.frameGeometry().size()
        left, top = area.left() + MARGIN, area.top() + MARGIN
        right, bottom = area.right() + 1 - size.width() - MARGIN, area.bottom() + 1 - size.height() - MARGIN
        cx, cy = area.center().x() - size.width() // 2, area.center().y() - size.height() // 2
        pos = {"top_left": (left, top), "top_right": (right, top), "bottom_left": (left, bottom),
               "bottom_right": (right, bottom), "center": (cx, cy), "top": (cx, top), "bottom": (cx, bottom),
               "left": (left, cy), "right": (right, cy)}[where]
        self._animate_to(w, QPoint(*pos), 420)
        return "placed"

    def glide(self, direction: str, speed: str = "normal") -> str:
        if direction not in DIRS:
            return "unknown direction"
        self.stop()
        self._ready()
        self._dir = DIRS[direction]
        self._speed = GLIDE_SPEED.get(speed, GLIDE_SPEED["normal"])
        self._glide_until = time.monotonic() + GLIDE_MAX_S
        self._last = time.monotonic()
        self._carry = 0.0
        self._glide.start()
        return "gliding"

    def _glide_step(self) -> None:
        w = self.target()
        if time.monotonic() > self._glide_until or not w.isVisible():
            self._glide.stop()
            return
        now = time.monotonic()  # distance by time, so a busy PC doesn't slow the glide down
        self._carry += self._speed * min(0.25, now - self._last)
        self._last = now
        step = int(self._carry)
        if step == 0:
            return
        self._carry -= step
        new = self.clamp(w, w.pos() + QPoint(self._dir[0] * step, self._dir[1] * step))
        if new == w.pos():  # reached the edge of the screen
            self._glide.stop()
            return
        w.move(new)

    @property
    def gliding(self) -> bool:
        return self._glide.isActive()

    def stop(self) -> None:
        self._glide.stop()
        if self._anim is not None:
            try:
                self._anim.stop()
            except RuntimeError:
                pass
            self._anim = None

    def resize(self, bigger: bool) -> str:
        w = self._ready()
        if w.minimumSize() == w.maximumSize():
            return "fixed"  # the mini orb has one size
        area = self._screen(w)
        f = 1.18 if bigger else 1 / 1.18
        size = QSize(int(w.width() * f), int(w.height() * f))
        size = size.expandedTo(w.minimumSize()).boundedTo(area.size() - QSize(2 * MARGIN, 2 * MARGIN))
        if size == w.size():
            return "limit"
        center = w.geometry().center()
        rect = QRect(QPoint(0, 0), size)
        rect.moveCenter(center)
        w.setGeometry(rect)
        w.move(self.clamp(w, w.pos()))
        return "resized"
