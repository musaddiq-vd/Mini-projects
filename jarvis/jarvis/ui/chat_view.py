"""The conversation view: message bubbles, tool chips, image thumbnails and search filtering."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, QUrl
from PyQt6.QtGui import QColor, QDesktopServices, QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QMenu, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

from jarvis.ui import theme
from jarvis.ui.fx import Typewriter, grow_in
from jarvis.ui.richtext import has_markup, to_html

MAX_CHAT_ROWS = 250
THUMB_WIDTH = 340


TOOL_ICONS = {
    "open": "🚀", "close": "✖", "volume": "🔊", "brightness": "☀", "media": "🎵", "play": "▶",
    "screenshot": "📸", "screen": "👁", "camera": "📷", "weather": "🌦", "search": "🔎", "news": "📰", "remind": "⏰",
    "remember": "🧠", "recall": "🧠", "memory": "🧠", "file": "📁", "folder": "📁", "whatsapp": "💬",
    "message": "💬", "email": "✉", "code": "💻", "type": "⌨", "key": "⌨", "clipboard": "📋",
    "power": "⏻", "wifi": "📶", "timer": "⏱", "focus": "🎯", "note": "📝", "routine": "🔁",
    "history": "🗂", "time": "🕒", "system": "📊", "undo": "↩", "sleep": "💤", "window": "🪟",
}


def tool_icon(name: str) -> str:
    for key, icon in TOOL_ICONS.items():
        if key in name:
            return icon
    return "⚡"


def pretty_tool(name: str) -> str:
    return name.replace("_", " ").capitalize()


class ImageThumb(QLabel):
    """Rounded, clickable thumbnail of an image file; click opens it."""

    def __init__(self, path: str, accent: str, parent=None):
        super().__init__(parent)
        self.path = path
        pix = QPixmap(path)
        if pix.isNull():
            self.setText(f"🖼  {Path(path).name}")
            return
        pix = pix.scaledToWidth(THUMB_WIDTH, Qt.TransformationMode.SmoothTransformation)
        rounded = QPixmap(pix.size())
        rounded.fill(Qt.GlobalColor.transparent)
        p = QPainter(rounded)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0, 0, pix.width(), pix.height()), 12, 12)
        p.setClipPath(clip)
        p.drawPixmap(0, 0, pix)
        p.setClipping(False)
        p.setPen(QColor(accent))
        p.drawRoundedRect(QRectF(0.5, 0.5, pix.width() - 1, pix.height() - 1), 12, 12)
        p.end()
        self.setPixmap(rounded)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"{Path(path).name}\nClick to open")

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and Path(self.path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.path))


class ChatView(QScrollArea):
    """Conversation as fading-in bubbles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(objectName="chatInner")
        self.col = QVBoxLayout(inner)
        self.col.setContentsMargins(4, 4, 10, 4)
        self.col.setSpacing(8)
        self.col.addStretch(1)
        self.setWidget(inner)
        self._anims: list[QPropertyAnimation] = []
        self._stick = True
        self.animate = True  # off while old history is put back
        self._last_day = None
        self.query = ""
        self.verticalScrollBar().rangeChanged.connect(self._on_range)
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)
        self.jump = QPushButton("↓  Latest", self.viewport(), objectName="jump")
        self.jump.setCursor(Qt.CursorShape.PointingHandCursor)
        self.jump.clicked.connect(self.scroll_to_end)
        self.jump.hide()
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._chat_menu)

    def scroll_to_end(self) -> None:
        self._stick = True
        bar = self.verticalScrollBar()
        anim = QPropertyAnimation(bar, b"value", self)
        anim.setDuration(350)
        anim.setEndValue(bar.maximum())
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start()
        self._scroll_anim = anim
        self.jump.hide()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.jump.adjustSize()
        self.jump.move(self.viewport().width() - self.jump.width() - 14,
                       self.viewport().height() - self.jump.height() - 10)

    def _on_scroll(self, value: int) -> None:
        self._stick = value >= self.verticalScrollBar().maximum() - 30
        if self._stick:
            self.jump.hide()

    def _chat_menu(self, pos) -> None:
        menu = QMenu(self)
        target = self.childAt(pos)
        bubble_text = None
        while target is not None and target is not self:
            if getattr(target, "_copy_text", None) is not None:
                bubble_text = target._copy_text
                break
            target = target.parentWidget()
        if bubble_text is not None:
            menu.addAction("Copy message").triggered.connect(lambda: QApplication.clipboard().setText(bubble_text))
        menu.addAction("Copy whole chat").triggered.connect(
            lambda: QApplication.clipboard().setText(self.toPlainText()))
        menu.exec(self.mapToGlobal(pos))

    def _day_separator(self, when: datetime) -> None:
        day = when.date()
        if day == self._last_day:
            return
        self._last_day = day
        today = datetime.now().date()
        text = "TODAY" if day == today else "YESTERDAY" if (today - day).days == 1 else when.strftime("%a, %d %b").upper()
        line = QLabel(f"──────   {text}   ──────", objectName="dayline")
        line.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row = self._add_row(line, "center", animate=False)
        row._kind = "day"

    def filter(self, query: str) -> int:
        """Show only rows containing the query; returns how many messages match."""
        self.query = query.strip().lower()
        matches = 0
        for i in range(1, self.col.count()):
            row = self.col.itemAt(i).widget()
            if row is None:
                continue
            if not self.query:
                row.setVisible(True)
                continue
            if getattr(row, "_kind", "") == "day":
                row.setVisible(False)
                continue
            hit = self.query in getattr(row, "_search", "").lower()
            row.setVisible(hit)
            matches += hit
        return matches

    def _on_range(self, _min: int, maximum: int) -> None:
        if self._stick:
            self.verticalScrollBar().setValue(maximum)

    def _add_row(self, widget: QWidget, align: str, text: str = "", animate: bool = True) -> QWidget:
        row = QWidget(objectName="chatRow")
        row._search = text
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        if align == "right":
            h.addStretch(1)
        h.addWidget(widget, 1 if align == "center" else 0)
        if align == "left":
            h.addStretch(1)
        self.col.addWidget(row)
        while self.col.count() > MAX_CHAT_ROWS + 1:
            old = self.col.takeAt(1).widget()
            if old:
                old.deleteLater()
        if self.query and self.query not in text.lower():
            row.hide()  # a search is open: new rows that don't match stay hidden
        if not self._stick and self.animate:
            self.jump.show()
            self.jump.raise_()
        if not (self.animate and animate):
            return row
        grow_in(row)
        effect = QGraphicsOpacityEffect(row)
        row.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(320)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda: (row.setGraphicsEffect(None), self._anims.remove(anim)))
        self._anims.append(anim)
        anim.start()
        self._stick = self._stick or not self.animate
        return row

    def _max_width(self, share: float) -> int:
        return max(240, int(self.viewport().width() * share))

    @staticmethod
    def _fit(label: QLabel, max_width: int, text: str | None = None) -> None:
        """Short text gets a snug label, long text wraps at max_width (QLabel can't decide alone)."""
        text = label.text() if text is None else text
        longest = max((label.fontMetrics().horizontalAdvance(line) for line in text.splitlines()), default=0)
        label.setMinimumWidth(min(longest + 6, max_width))

    def add_message(self, who: str, text: str, mine: bool, when: datetime | None = None,
                    typewriter: bool = False) -> None:
        when = when or datetime.now()
        self._day_separator(when)
        bubble = QFrame(objectName="bubbleMe" if mine else "bubbleAi")
        bubble._copy_text = text
        v = QVBoxLayout(bubble)
        v.setContentsMargins(12, 8, 12, 10)
        v.setSpacing(2)
        stamp = when.strftime("%I:%M %p").lstrip("0")
        if when.date() != datetime.now().date():
            stamp = when.strftime("%d %b, ") + stamp
        name = QLabel(f"{who.upper()}   ·   {stamp}", objectName="who")
        if mine:
            name.setAlignment(Qt.AlignmentFlag.AlignRight)
        v.addWidget(name)
        msg = QLabel(objectName="msg")
        msg.setWordWrap(True)
        msg.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)  # the chat's own menu (Copy) shows
        html_text = to_html(text, theme.readable(theme.ACCENT)) if has_markup(text) else None
        if html_text is not None:
            msg.setTextFormat(Qt.TextFormat.RichText)
            msg.setText(html_text)
            msg.setOpenExternalLinks(True)
            msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        else:
            msg.setTextFormat(Qt.TextFormat.PlainText)
            msg.setText(text)
            msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(msg)
        width = self._max_width(0.78)
        self._fit(msg, width - 26, text)
        bubble.setMaximumWidth(width)
        bubble.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self._add_row(bubble, "right" if mine else "left", text=f"{who} {text}")
        if typewriter and self.animate:
            msg._typewriter = Typewriter(msg, text, final_html=html_text)

    def add_tool(self, name: str, args: dict, result: dict) -> None:
        ok = "error" not in result
        detail = str(result.get("error") or result.get("result") or "")
        if not result.get("error"):
            first = detail.split(". ")[0]
            if first != detail:
                detail = first + "."  # first sentence only; the rest is for the model
        if len(detail) > 90:
            detail = detail[:88].rsplit(" ", 1)[0] + "…"
        args_text = ", ".join(f"{k}={str(v)[:30]}" for k, v in args.items())
        line = f"{tool_icon(name)}  {pretty_tool(name)}" + (f"  ({args_text})" if args_text else "")
        line += ("  ✓" if ok else "  ✗") + (f"  {detail}" if detail else "")
        chip = QLabel(line, objectName="toolOk" if ok else "toolErr")
        chip.setTextFormat(Qt.TextFormat.PlainText)
        chip.setWordWrap(True)
        width = self._max_width(0.9)
        self._fit(chip, width - 22)
        chip.setMaximumWidth(width)
        self._add_row(chip, "left", text=line)

    def add_image(self, path: str, caption: str, accent: str) -> None:
        box = QFrame(objectName="bubbleAi")
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        v.addWidget(ImageThumb(path, accent))
        cap = QLabel(caption, objectName="muted")
        cap.setTextFormat(Qt.TextFormat.PlainText)
        v.addWidget(cap)
        self._add_row(box, "left", text=caption)

    def add_system(self, text: str) -> None:
        label = QLabel(text, objectName="muted")
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._add_row(label, "center", text=text)

    def toPlainText(self) -> str:  # noqa: N802 - same name as QTextBrowser
        """All shown text, top to bottom."""
        def full(label: QLabel) -> str:  # a message still typing itself out counts in full
            writer = getattr(label, "_typewriter", None)
            if writer:
                return writer.text
            parent = label.parentWidget()
            if label.textFormat() == Qt.TextFormat.RichText and getattr(parent, "_copy_text", None) is not None:
                return parent._copy_text
            return label.text()

        return "\n".join(full(label) for label in self.widget().findChildren(QLabel))

    def clear(self) -> None:
        self._last_day = None
        while self.col.count() > 1:
            w = self.col.takeAt(1).widget()
            if w:
                w.deleteLater()
