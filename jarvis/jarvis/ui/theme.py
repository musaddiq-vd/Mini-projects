"""Colours and the Qt stylesheet, built from one accent colour."""

from __future__ import annotations

from PyQt6.QtGui import QColor

BG = "#0a0f14"
PANEL = "#101820"
PANEL_2 = "#16212b"
TEXT = "#d8e6ef"
MUTED = "#6f8594"
WARN = "#ffb547"
DANGER = "#ff5c6c"

STATE_LABELS = {
    "connecting": "CONNECTING",
    "listening": "LISTENING",
    "speaking": "SPEAKING",
    "thinking": "WORKING",
    "sleeping": "SLEEPING",
    "muted": "MIC MUTED",
    "offline": "OFFLINE",
}


def valid_color(value: str, fallback: str = "#00d4ff") -> str:
    color = QColor(value)
    return color.name() if color.isValid() else fallback


def state_color(state: str, accent: str) -> QColor:
    if state == "thinking":
        return QColor(WARN)
    if state == "muted":
        return QColor(DANGER)
    if state in {"sleeping", "connecting", "offline"}:
        return QColor(MUTED)
    color = QColor(accent)
    return color.lighter(125) if state == "speaking" else color


def stylesheet(accent: str) -> str:
    a = QColor(accent)
    soft = f"rgba({a.red()},{a.green()},{a.blue()},40)"
    mid = f"rgba({a.red()},{a.green()},{a.blue()},110)"
    return f"""
    QWidget {{ background: {BG}; color: {TEXT}; font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 13px; }}
    QFrame#panel {{ background: {PANEL}; border: 1px solid {soft}; border-radius: 14px; }}
    QLabel#title {{ color: {accent}; font-size: 20px; font-weight: 700; letter-spacing: 4px; background: transparent; }}
    QLabel#state {{ color: {accent}; font-size: 12px; font-weight: 600; letter-spacing: 3px; background: transparent; }}
    QLabel#muted {{ color: {MUTED}; font-size: 11px; background: transparent; }}
    QLabel#section {{ color: {MUTED}; font-size: 11px; font-weight: 600; letter-spacing: 2px; background: transparent; }}
    QTextBrowser {{ background: transparent; border: none; selection-background-color: {mid}; }}
    QLineEdit, QSpinBox, QComboBox {{
        background: {PANEL_2}; border: 1px solid {soft}; border-radius: 8px; padding: 8px 10px; color: {TEXT};
    }}
    QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border: 1px solid {accent}; }}
    QComboBox QAbstractItemView {{ background: {PANEL_2}; selection-background-color: {mid}; }}
    QPushButton {{
        background: {PANEL_2}; border: 1px solid {soft}; border-radius: 8px; padding: 8px 14px; color: {TEXT};
    }}
    QPushButton:hover {{ border: 1px solid {accent}; }}
    QPushButton:checked {{ background: {mid}; border: 1px solid {accent}; }}
    QPushButton#primary {{ background: {accent}; color: {BG}; font-weight: 700; border: none; }}
    QPushButton#danger {{ background: {DANGER}; color: white; font-weight: 700; border: none; }}
    QFrame#banner {{ background: rgba(255,181,71,30); border: 1px solid {WARN}; border-radius: 10px; }}
    QTabWidget::pane {{ border: 1px solid {soft}; border-radius: 10px; background: {PANEL}; }}
    QTabBar::tab {{ background: {PANEL_2}; color: {MUTED}; padding: 8px 14px; border-radius: 6px; margin: 2px; }}
    QTabBar::tab:selected {{ color: {BG}; background: {accent}; }}
    QCheckBox {{ background: transparent; spacing: 8px; }}
    QTableWidget {{ background: {PANEL_2}; gridline-color: {soft}; border: none; border-radius: 8px; }}
    QHeaderView::section {{ background: {PANEL}; color: {MUTED}; border: none; padding: 6px; }}
    QScrollBar:vertical {{ background: transparent; width: 8px; }}
    QScrollBar::handle:vertical {{ background: {mid}; border-radius: 4px; min-height: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    """
