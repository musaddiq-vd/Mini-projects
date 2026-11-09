"""Draw the Jarvis app icon (a glowing orb with a face) as installer/assets/jarvis.ico and jarvis.png."""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"


def draw(size: int = 256):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import QPointF, QRectF, Qt
    from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QRadialGradient
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication(sys.argv)
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QPointF(size / 2, size / 2)
    r = size * 0.46
    accent, second = QColor("#00d4ff"), QColor("#6a5cff")
    glow = QRadialGradient(c, r)
    glow.setColorAt(0.0, QColor(10, 18, 28))
    glow.setColorAt(0.78, QColor(8, 14, 22))
    glow.setColorAt(1.0, QColor(8, 14, 22, 0))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(glow)
    p.drawEllipse(c, r, r)
    ring = QPen(accent, size * 0.045)
    p.setPen(ring)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(c, r * 0.86, r * 0.86)
    p.setPen(QPen(second, size * 0.03))
    p.drawArc(QRectF(c.x() - r * 0.86, c.y() - r * 0.86, r * 1.72, r * 1.72), 40 * 16, 110 * 16)
    core = QRadialGradient(QPointF(c.x() - r * 0.15, c.y() - r * 0.2), r * 0.7)
    core.setColorAt(0, accent.lighter(150))
    core.setColorAt(0.55, accent)
    core.setColorAt(1, accent.darker(220))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(core)
    p.drawEllipse(c, r * 0.62, r * 0.62)
    p.setBrush(QColor(4, 16, 24))
    for side in (-1, 1):  # eyes
        p.drawRoundedRect(QRectF(c.x() + side * r * 0.22 - r * 0.07, c.y() - r * 0.2, r * 0.14, r * 0.26),
                          r * 0.06, r * 0.06)
    p.setPen(QPen(QColor(4, 16, 24), size * 0.028, cap=Qt.PenCapStyle.RoundCap))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawArc(QRectF(c.x() - r * 0.18, c.y() + r * 0.02, r * 0.36, r * 0.24), 200 * 16, 140 * 16)  # smile
    p.end()
    return img


def main() -> None:
    from PIL import Image

    ASSETS.mkdir(parents=True, exist_ok=True)
    png = ASSETS / "jarvis.png"
    draw(256).save(str(png))
    Image.open(png).save(ASSETS / "jarvis.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                                                         (128, 128), (256, 256)])
    print("icon:", ASSETS / "jarvis.ico")


if __name__ == "__main__":
    main()
