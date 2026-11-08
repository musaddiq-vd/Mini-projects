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
OK = "#3ddc97"

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
    QWidget {{ background: transparent; color: {TEXT}; font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 13px; }}
    QMainWindow, QDialog, QMessageBox, QMenu {{ background: {BG}; }}
    QMenu {{ border: 1px solid {soft}; padding: 6px; border-radius: 8px; }}
    QMenu::item {{ padding: 6px 18px; border-radius: 6px; }}
    QMenu::item:selected {{ background: {mid}; }}
    QToolTip {{ background: {PANEL_2}; color: {TEXT}; border: 1px solid {soft}; padding: 6px; border-radius: 6px; }}
    QFrame#panel {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(20,32,44,190), stop:1 rgba(10,17,24,200));
        border: 1px solid {soft}; border-top: 1px solid {mid}; border-radius: 16px;
    }}
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
    QPushButton:hover {{ border: 1px solid {accent}; background: {soft}; }}
    QPushButton:pressed {{ background: {mid}; padding-top: 9px; padding-bottom: 7px; }}
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
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
    QScrollBar::handle:vertical:hover {{ background: {accent}; }}
    QWidget#root {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #070b10, stop:1 #0d1620); }}
    QScrollArea, QWidget#chatInner, QWidget#chatRow {{ background: transparent; border: none; }}
    QFrame#bubbleAi {{ background: {PANEL_2}; border: 1px solid {soft}; border-radius: 14px;
                       border-top-left-radius: 4px; }}
    QFrame#bubbleMe {{ background: {mid}; border: 1px solid {accent}; border-radius: 14px;
                       border-top-right-radius: 4px; }}
    QLabel#who {{ color: {accent}; font-size: 10px; font-weight: 700; letter-spacing: 1px; background: transparent; }}
    QLabel#msg {{ color: {TEXT}; font-size: 14px; background: transparent; }}
    QLabel#toolOk, QLabel#toolErr {{ font-family: Consolas, monospace; font-size: 11px; padding: 3px 10px;
                                     border-radius: 9px; background: rgba(255,255,255,8); }}
    QLabel#toolOk {{ color: {OK}; border: 1px solid rgba(61,220,151,70); }}
    QLabel#toolErr {{ color: {WARN}; border: 1px solid rgba(255,181,71,90); }}
    QLabel#caption {{ color: {TEXT}; font-size: 13px; font-style: italic; background: transparent; }}
    QPushButton#chip {{ background: transparent; border: 1px solid {soft}; border-radius: 13px;
                        padding: 4px 11px; font-size: 12px; color: {MUTED}; }}
    QPushButton#chip:hover {{ color: {TEXT}; border: 1px solid {accent}; background: {soft}; }}
    QPushButton#small {{ background: transparent; border: none; color: {MUTED}; padding: 2px 6px; }}
    QPushButton#small:hover {{ color: {accent}; }}
    """
