"""A layout that places widgets in a row and wraps them to the next line when there's no room (like words).

Used for chip rows, so a narrow window shows them on two lines instead of forcing the window wide.
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import QLayout, QLayoutItem, QSizePolicy, QWidget


class FlowLayout(QLayout):
    def __init__(self, parent: QWidget | None = None, spacing: int = 6, center: bool = False):
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._spacing = spacing
        self._center = center
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i: int) -> QLayoutItem | None:  # noqa: N802
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i: int) -> QLayoutItem | None:  # noqa: N802
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802
        w = h = 0
        for item in self._visible():
            s = item.sizeHint()
            w += s.width() + self._spacing
            h = max(h, s.height())
        m = self.contentsMargins()
        return QSize(max(0, w - self._spacing) + m.left() + m.right(), h + m.top() + m.bottom())

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._visible():
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _visible(self) -> list[QLayoutItem]:
        return [i for i in self._items if not (i.widget() and i.widget().isHidden())]

    def _arrange(self, rect: QRect, apply: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        lines: list[list[tuple[QLayoutItem, QSize]]] = [[]]
        x = 0
        for item in self._visible():
            hint = item.sizeHint()
            if lines[-1] and x + hint.width() > area.width():  # wrap
                lines.append([])
                x = 0
            lines[-1].append((item, hint))
            x += hint.width() + self._spacing
        y = area.y()
        for line in lines:
            if not line:
                continue
            used = sum(h.width() for _, h in line) + self._spacing * (len(line) - 1)
            x = area.x() + (max(0, area.width() - used) // 2 if self._center else 0)
            line_h = max(h.height() for _, h in line)
            for item, hint in line:
                if apply:
                    item.setGeometry(QRect(QPoint(x, y + (line_h - hint.height()) // 2), hint))
                x += hint.width() + self._spacing
            y += line_h + self._spacing
        return max(0, y - self._spacing) - rect.y() + m.bottom()


def flow_widget(spacing: int = 6, center: bool = False) -> tuple[QWidget, FlowLayout]:
    w = QWidget()
    lay = FlowLayout(w, spacing, center)
    policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    policy.setHeightForWidth(True)
    w.setSizePolicy(policy)
    return w, lay
