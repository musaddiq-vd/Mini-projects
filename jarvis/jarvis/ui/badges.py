"""Achievements gallery: every badge, unlocked ones in colour with the date, locked ones with progress."""

from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from jarvis.ui import theme


def _badge_card(badge, count: int, unlocked_at: float | None) -> QFrame:
    card = QFrame(objectName="cardGood" if unlocked_at else "card")
    v = QVBoxLayout(card)
    v.setContentsMargins(12, 10, 12, 10)
    v.setSpacing(4)
    top = QHBoxLayout()
    icon = QLabel(badge.icon if unlocked_at else "🔒")
    icon.setStyleSheet("font-size: 26px; background: transparent;")
    top.addWidget(icon)
    title = QLabel(badge.title, objectName="h2")
    if not unlocked_at:
        title.setStyleSheet(f"color: {theme.MUTED}; background: transparent;")
    top.addWidget(title, 1)
    v.addLayout(top)
    desc = QLabel(badge.desc, objectName="muted")
    desc.setWordWrap(True)
    v.addWidget(desc)
    if unlocked_at:
        v.addWidget(QLabel(f"Unlocked {datetime.fromtimestamp(unlocked_at):%d %b %Y}", objectName="good"))
    else:
        bar = QProgressBar()
        bar.setMaximum(badge.goal)
        bar.setValue(count)
        bar.setTextVisible(False)
        v.addWidget(bar)
        v.addWidget(QLabel(f"{count} / {badge.goal}", objectName="muted"))
    return card


class BadgesPanel(QFrame):
    def __init__(self, parent: QWidget, achievements):
        super().__init__(parent)
        self.setObjectName("badgesPanel")
        self.achievements = achievements
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(18, 16, 18, 16)
        self.hide()

    def _build(self) -> None:
        while self.v.count():
            item = self.v.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    w = item.layout().takeAt(0).widget()
                    if w:
                        w.deleteLater()
        self.setStyleSheet(f"QFrame#badgesPanel {{ background: {theme.BG}; border: 1px solid {theme.ACCENT}; "
                           "border-radius: 16px; }")
        items = self.achievements.progress()
        got = sum(1 for _, _, at in items if at)
        head = QHBoxLayout()
        head.addWidget(QLabel("ACHIEVEMENTS", objectName="section"))
        head.addStretch(1)
        head.addWidget(QLabel(f"🏆 {got} of {len(items)} unlocked  ·  🔥 {self.achievements.streak()}-day streak",
                              objectName="accent"))
        close = QPushButton("✕", objectName="small")
        close.clicked.connect(self.hide)
        head.addWidget(close)
        self.v.addLayout(head)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setSpacing(10)
        # unlocked first, then the closest to unlocking
        items.sort(key=lambda it: (it[2] is None, -(it[1] / it[0].goal)))
        for i, (badge, count, at) in enumerate(items):
            grid.addWidget(_badge_card(badge, count, at), i // 3, i % 3)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        self.v.addWidget(scroll, 1)

    def toggle(self) -> None:
        if self.isVisible():
            self.hide()
            return
        self._build()
        self.setGeometry(self.parentWidget().rect().adjusted(8, 8, -8, -8))
        self.show()
        self.raise_()
