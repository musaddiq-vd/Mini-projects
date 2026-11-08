"""'What can I say?' - a panel of example commands by topic, and the welcome card for an empty chat.

Every example is clickable: it is sent to Jarvis exactly as if you had typed it.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt
from PyQt6.QtWidgets import (
    QFrame, QGraphicsOpacityEffect, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)


def _chip(text: str, send: Callable[[str], None], quoted: bool = False) -> QPushButton:
    b = QPushButton(f"“{text}”" if quoted else text, objectName="chip")
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setToolTip(text)
    b.setMinimumWidth(0)  # long examples shrink with the window instead of pushing it wider
    b.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    b.setStyleSheet("text-align: left;")
    b.clicked.connect(lambda _, t=text: send(t))
    return b

TOPICS: list[tuple[str, str, list[str]]] = [
    ("🚀", "Apps & windows", ["Open Spotify", "Close YouTube", "Minimize everything", "Switch to VS Code"]),
    ("🕹", "Move by voice", ["Apni window right corner me le jao", "Dheere dheere neeche chalte raho",
                           "Is page ko neeche scroll karte raho", "Ruko", "Chrome ko left half me lagao"]),
    ("🔊", "System", ["Volume 40 percent", "Brightness 70", "Turn on dark mode", "How is my battery?"]),
    ("👁", "Screen & files", ["What's on my screen?", "Is error ko theek karo", "Screenshot lo aur kholo",
                             "Read my latest download"]),
    ("🌦", "Web & news", ["Weather today", "Top news in two lines", "Search the web for Python 3.14 features",
                         "Flights from Pune to Delhi next Friday"]),
    ("⏱", "Timers & reminders", ["5 minute ka timer lagao", "Remind me at 6:30 pm to call mom",
                                 "Roz subah 9 baje vitamins yaad dilana", "Focus mode 25 minutes"]),
    ("📝", "Notes & routines", ["Milk aur bread list mein daal do", "Meri list batao",
                               "Save a routine called work mode: open VS Code and Chrome", "work mode"]),
    ("💬", "Messages", ["WhatsApp mom: I'll be late", "Draft an email to my boss about leave tomorrow"]),
    ("🧠", "Memory & chat", ["Remember that my city is Pune", "What do you know about me?",
                            "Kal maine kya poocha tha?", "Aaj ka recap do", "Undo"]),
    ("💻", "Code", ["Write a Python snake game in a folder called snake", "Explain main.py"]),
    ("😄", "Fun", ["Tell me a joke", "Thank you Jarvis!", "Sing me a short song about chai"]),
]


def _card(icon: str, title: str, examples: list[str], send: Callable[[str], None]) -> QFrame:
    card = QFrame(objectName="card")
    v = QVBoxLayout(card)
    v.setContentsMargins(12, 10, 12, 12)
    v.setSpacing(6)
    v.addWidget(QLabel(f"{icon}  {title}", objectName="h2"))
    for ex in examples:
        v.addWidget(_chip(ex, send, quoted=True))
    v.addStretch(1)
    return card


SHORTCUTS = [
    ("Ctrl+Alt+J", "Call Jarvis from any app"), ("Ctrl+K", "Type a message"), ("Ctrl+M", "Mute the mic"),
    ("Ctrl+Space", "Sleep / wake"), ("Ctrl+P", "Planner"), ("Ctrl+F", "Search the chat"), ("F1", "This help"),
    ("Ctrl+Shift+M", "Mini mode"), ("Ctrl+L", "Clear the chat"), ("Ctrl+,", "Settings"), ("↑ / ↓", "Earlier messages"),
    ("Esc", "Close panels"),
]


def _shortcuts_card() -> QFrame:
    card = QFrame(objectName="card")
    v = QVBoxLayout(card)
    v.setContentsMargins(12, 10, 12, 12)
    v.addWidget(QLabel("⌨  Keyboard shortcuts", objectName="h2"))
    grid = QGridLayout()
    grid.setHorizontalSpacing(14)
    grid.setVerticalSpacing(6)
    for i, (keys, what) in enumerate(SHORTCUTS):
        key = QLabel(keys)
        key.setStyleSheet("font-family: Consolas, monospace; font-weight: 700; padding: 2px 8px; "
                          "border: 1px solid rgba(127,127,127,90); border-radius: 6px;")
        key.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        grid.addWidget(key, i // 2, (i % 2) * 2)
        grid.addWidget(QLabel(what, objectName="muted"), i // 2, (i % 2) * 2 + 1)
    grid.setColumnStretch(1, 1)
    grid.setColumnStretch(3, 1)
    v.addLayout(grid)
    return card


class HelpPanel(QFrame):
    """Overlay over the conversation. Esc, the ✕ or any example closes it."""

    def __init__(self, parent: QWidget, send: Callable[[str], None]):
        super().__init__(parent)
        self.setObjectName("helpPanel")
        from jarvis.ui import theme

        self.setStyleSheet(f"QFrame#helpPanel {{ background: {theme.BG}; border: 1px solid {theme.ACCENT}; "
                           "border-radius: 16px; }")
        self.send = send
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 16)
        head = QHBoxLayout()
        head.addWidget(QLabel("WHAT CAN I SAY?", objectName="section"))
        head.addStretch(1)
        head.addWidget(QLabel("Talk or type in any language · click an example to try it", objectName="muted"))
        close = QPushButton("✕", objectName="small")
        close.clicked.connect(self.close_panel)
        head.addWidget(close)
        v.addLayout(head)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setSpacing(10)
        grid.setContentsMargins(0, 0, 12, 0)  # room for the scrollbar
        for i, (icon, title, examples) in enumerate(TOPICS):
            grid.addWidget(_card(icon, title, examples, self._pick), i // 2, i % 2)
        grid.addWidget(_shortcuts_card(), len(TOPICS) // 2 + 1, 0, 1, 2)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        v.addWidget(scroll, 1)
        self.hide()

    def _pick(self, text: str) -> None:
        self.close_panel()
        self.send(text)

    def open_panel(self) -> None:
        from jarvis.ui import theme  # follow day/night changes

        self.setStyleSheet(f"QFrame#helpPanel {{ background: {theme.BG}; border: 1px solid {theme.ACCENT}; "
                           "border-radius: 16px; }")
        self.setGeometry(self.parentWidget().rect().adjusted(8, 8, -8, -8))
        self.show()
        self.raise_()
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(220)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda: self.setGraphicsEffect(None))
        anim.start()
        self._anim = anim

    def close_panel(self) -> None:
        self.hide()

    def toggle(self) -> None:
        self.close_panel() if self.isVisible() else self.open_panel()


def welcome_card(name: str, user: str, send: Callable[[str], None]) -> QFrame:
    """Shown in an empty chat: a greeting and a few things to try."""
    card = QFrame(objectName="card")
    v = QVBoxLayout(card)
    v.setContentsMargins(18, 16, 18, 16)
    v.setSpacing(8)
    hello = f"Hi{' ' + user.split()[0].title() if user else ''}! I'm {name}."
    v.addWidget(QLabel(f"👋  {hello}", objectName="h1"))
    sub = QLabel("Talk to me in English, Hindi, Urdu or Hinglish - or type below. A few things to try:",
                 objectName="muted")
    sub.setWordWrap(True)
    v.addWidget(sub)
    grid = QGridLayout()
    grid.setSpacing(6)
    for i, ex in enumerate(("What's the weather today?", "5 minute ka timer lagao", "What's on my screen?",
                            "Tell me a joke")):
        grid.addWidget(_chip(ex, send), i // 2, i % 2)
    v.addLayout(grid)
    tip = QLabel("Press F1 any time to see everything I can do.", objectName="muted")
    tip.setWordWrap(True)
    v.addWidget(tip)
    return card
