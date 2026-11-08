"""Which monitor things appear on, for setups with more than one screen.

Pop-ups follow a setting: the screen with the mouse (where you are looking), the main screen, or the screen with
the Jarvis window. The desktop widget remembers its own screen by name, and falls back to the main screen when
that monitor is unplugged.
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtGui import QCursor, QGuiApplication, QScreen

POPUP_MODES = ("mouse", "main", "jarvis")


def screens() -> list[QScreen]:
    return list(QGuiApplication.screens())


def main_screen() -> QScreen | None:
    return QGuiApplication.primaryScreen()


def by_name(name: str) -> QScreen | None:
    """The screen with this name; the main screen when it isn't connected (or name is empty)."""
    for s in screens():
        if name and s.name() == name:
            return s
    return main_screen()


def at_point(point: QPoint) -> QScreen | None:
    return QGuiApplication.screenAt(point) or main_screen()


def for_popups(mode: str, jarvis_window=None) -> QScreen | None:
    if mode == "main":
        return main_screen()
    if mode == "jarvis" and jarvis_window is not None:
        try:
            if jarvis_window.isVisible():
                return jarvis_window.screen() or main_screen()
        except RuntimeError:
            pass
    return at_point(QCursor.pos())


def label(s: QScreen, index: int) -> str:
    g = s.geometry()
    main = " (main)" if s is main_screen() else ""
    return f"Screen {index + 1}{main} · {g.width()}×{g.height()}"


def area(s: QScreen | None) -> QRect | None:
    return s.availableGeometry() if s is not None else None


def visible_anywhere(point: QPoint, margin: int = 50) -> bool:
    """Is this point on (or just next to) any connected screen?"""
    return any(s.availableGeometry().adjusted(-margin, -margin, margin, margin).contains(point) for s in screens())
