"""Mini mode: a small floating orb that stays on top of other windows.

Drag it anywhere, double-click to go back to the full window, right-click for a menu.
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QPointF, QRectF, QSettings, Qt
from PyQt6.QtGui import QAction, QColor, QFont, QPainter, QRadialGradient
from PyQt6.QtWidgets import QApplication, QLabel, QMenu, QVBoxLayout, QWidget

from jarvis.ui.orb import Orb


class OrbGroup:
    """Sends every orb call to all orbs (the big one and the mini one)."""

    def __init__(self, *orbs: Orb):
        self.orbs = list(orbs)

    def __getattr__(self, name):
        def call(*args, **kwargs):
            for orb in self.orbs:
                getattr(orb, name)(*args, **kwargs)
        return call


class MiniWindow(QWidget):
    SIZE = (230, 290)

    def __init__(self, accent: str, face: bool, on_restore, actions: list[tuple[str, object]]):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setObjectName("mini")
        self.setStyleSheet("QWidget#mini, QWidget#mini QWidget { background: transparent; }")
        self.on_restore = on_restore
        self.actions = actions
        self._drag: QPoint | None = None
        self.accent = accent
        self.setFixedSize(*self.SIZE)
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 8)
        v.setSpacing(0)
        self.orb = Orb(accent, face=face)
        self.orb.setMinimumSize(200, 220)
        self.orb.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        v.addWidget(self.orb, 1)
        self.caption = QLabel("")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setWordWrap(True)
        self.caption.setFixedHeight(40)
        self.caption.setFont(QFont("Segoe UI", 9))
        self.caption.setStyleSheet("color:#d8e6ef; background: transparent;")
        self.caption.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        v.addWidget(self.caption)
        self.setToolTip("Drag to move · double-click to open Jarvis · right-click for options")

    def set_caption(self, text: str) -> None:
        self.caption.setText(text[-90:] if len(text) > 90 else text)

    def set_accent(self, accent: str) -> None:
        self.accent = accent
        self.orb.set_accent(accent)
        self.update()

    def show_at_saved_place(self) -> None:
        pos = QSettings("MusaddiqKhan", "Jarvis").value("mini_pos")
        screen = QApplication.primaryScreen().availableGeometry()
        if isinstance(pos, QPoint) and screen.adjusted(-50, -50, 50, 50).contains(pos):
            self.move(pos)
        else:
            self.move(screen.right() - self.width() - 24, screen.bottom() - self.height() - 24)
        self.show()
        self.raise_()

    # -- painting: a soft dark disc behind the orb so it reads on any wallpaper ----------
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = QPointF(self.width() / 2, 118)
        grad = QRadialGradient(center, 112)
        grad.setColorAt(0.0, QColor(10, 15, 20, 235))
        grad.setColorAt(0.8, QColor(10, 15, 20, 200))
        grad.setColorAt(1.0, QColor(10, 15, 20, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawEllipse(center, 112, 112)
        if self.caption.text():
            p.setBrush(QColor(10, 15, 20, 210))
            p.drawRoundedRect(QRectF(6, self.height() - 50, self.width() - 12, 44), 12, 12)
        p.end()

    # -- mouse ------------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag is not None:
            QSettings("MusaddiqKhan", "Jarvis").setValue("mini_pos", self.pos())
        self._drag = None

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.on_restore()

    def closeEvent(self, event) -> None:  # noqa: N802
        if event.spontaneous():  # Alt+F4 by the user: go back to the full window instead
            event.ignore()
            self.on_restore()
        else:
            event.accept()  # the app is quitting: never block it

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        menu = QMenu(self)
        for text, fn in [("Open full window", self.on_restore), *self.actions]:
            act = QAction(text, menu)
            act.triggered.connect(fn)
            menu.addAction(act)
        menu.exec(event.globalPos())
