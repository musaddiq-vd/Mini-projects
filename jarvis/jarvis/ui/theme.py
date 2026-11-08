"""Themes: a palette (dark or light) + an accent colour -> the Qt stylesheet.

`apply(theme_name, mode, accent)` switches the whole app. Widgets read colours from this module
at paint time (theme.TEXT, theme.P.panel2 ...), so a switch takes effect everywhere at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from PyQt6.QtGui import QColor

WARN = "#ffb547"
DANGER = "#ff5c6c"
OK = "#3ddc97"


@dataclass(frozen=True)
class Palette:
    light: bool
    bg: str
    bg_top: str
    bg_bottom: str
    panel_top: str  # rgba(...) strings for the glass panels
    panel_bottom: str
    panel2: str
    text: str
    muted: str
    track: QColor  # empty part of meters / rings
    toast_bg: str


DARK = Palette(False, "#0a0f14", "#05080c", "#0b141d", "rgba(20,32,44,190)", "rgba(10,17,24,200)", "#16212b",
               "#d8e6ef", "#6f8594", QColor(255, 255, 255, 22), "rgba(12,20,28,235)")
LIGHT = Palette(True, "#eef3f7", "#f7fafc", "#e2eaf1", "rgba(255,255,255,215)", "rgba(244,248,251,225)", "#ffffff",
                "#13202b", "#5b6e7f", QColor(0, 30, 60, 26), "rgba(255,255,255,245)")

# name -> (accent, second aurora colour)
THEMES: dict[str, tuple[str, str]] = {
    "Jarvis Blue": ("#00d4ff", "#6a5cff"),
    "Neon Pink": ("#ff3df2", "#00e5ff"),
    "Emerald": ("#00e6a0", "#00a2ff"),
    "Sunset": ("#ff8a3d", "#ff3d81"),
    "Royal Purple": ("#8b7bff", "#ff5cd6"),
    "Mono": ("#9fb3c4", "#5d7385"),
}
MODES = ("dark", "light", "auto")

# current state (updated by apply)
P: Palette = DARK
BG, PANEL_2, TEXT, MUTED = DARK.bg, DARK.panel2, DARK.text, DARK.muted
SECOND = "#6a5cff"
ACCENT = "#00d4ff"

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


def is_day(now: datetime | None = None) -> bool:
    return 7 <= (now or datetime.now()).hour < 19


def resolve_mode(mode: str, now: datetime | None = None) -> str:
    if mode == "auto":
        return "light" if is_day(now) else "dark"
    return mode if mode in ("dark", "light") else "dark"


def accent_for(theme_name: str, custom: str = "") -> str:
    """The theme's accent, unless the user picked their own colour (one that isn't a theme accent)."""
    theme_accents = {a for a, _ in THEMES.values()}
    if custom and QColor(custom).isValid() and valid_color(custom) not in theme_accents:
        return valid_color(custom)
    return THEMES.get(theme_name, THEMES["Jarvis Blue"])[0]


def readable(accent: str) -> str:
    """Accent tuned for text on the current background (darker on light themes)."""
    c = QColor(accent)
    if P.light and c.lightness() > 110:
        c = c.darker(165)
    return c.name()


def apply(theme_name: str, mode: str, accent: str) -> str:
    """Switch palette; returns the stylesheet to set on the QApplication."""
    global P, BG, PANEL_2, TEXT, MUTED, SECOND, ACCENT
    P = LIGHT if resolve_mode(mode) == "light" else DARK
    BG, PANEL_2, TEXT, MUTED = P.bg, P.panel2, P.text, P.muted
    SECOND = THEMES.get(theme_name, THEMES["Jarvis Blue"])[1]
    ACCENT = valid_color(accent)
    return stylesheet(ACCENT)


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
    soft = f"rgba({a.red()},{a.green()},{a.blue()},{70 if P.light else 40})"
    mid = f"rgba({a.red()},{a.green()},{a.blue()},{75 if P.light else 110})"
    strong = readable(accent)
    on_accent = "#ffffff" if P.light else BG
    good_text = "#11804f" if P.light else OK
    warn_text = "#9a5b00" if P.light else WARN
    return f"""
    QWidget {{ background: transparent; color: {TEXT}; font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 13px; }}
    QMainWindow, QDialog, QMessageBox, QMenu {{ background: {BG}; }}
    QMenu {{ border: 1px solid {soft}; padding: 6px; border-radius: 8px; }}
    QMenu::item {{ padding: 6px 18px; border-radius: 6px; }}
    QMenu::item:selected {{ background: {mid}; }}
    QToolTip {{ background: {PANEL_2}; color: {TEXT}; border: 1px solid {soft}; padding: 6px; border-radius: 6px; }}
    QFrame#panel {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {P.panel_top}, stop:1 {P.panel_bottom});
        border: 1px solid {soft}; border-top: 1px solid {mid}; border-radius: 16px;
    }}
    QLabel#title {{ color: {strong}; font-size: 20px; font-weight: 700; letter-spacing: 4px; background: transparent; }}
    QLabel#state {{ color: {strong}; font-size: 12px; font-weight: 600; letter-spacing: 3px; background: transparent; }}
    QLabel#muted {{ color: {MUTED}; font-size: 11px; background: transparent; }}
    QLabel#section {{ color: {MUTED}; font-size: 11px; font-weight: 600; letter-spacing: 2px; background: transparent; }}
    QLabel#h1 {{ color: {TEXT}; font-size: 18px; font-weight: 700; background: transparent; }}
    QLabel#h2 {{ color: {TEXT}; font-size: 14px; font-weight: 600; background: transparent; }}
    QLabel#accent {{ color: {strong}; font-weight: 700; background: transparent; }}
    QTextBrowser {{ background: transparent; border: none; selection-background-color: {mid}; }}
    QLineEdit, QSpinBox, QComboBox, QTextEdit {{
        background: {PANEL_2}; border: 1px solid {soft}; border-radius: 8px; padding: 8px 10px; color: {TEXT};
    }}
    QLineEdit:focus, QSpinBox:focus, QComboBox:focus, QTextEdit:focus {{ border: 1px solid {accent}; }}
    QComboBox QAbstractItemView {{ background: {PANEL_2}; color: {TEXT}; selection-background-color: {mid}; }}
    QPushButton {{
        background: {PANEL_2}; border: 1px solid {soft}; border-radius: 8px; padding: 8px 14px; color: {TEXT};
    }}
    QPushButton:hover {{ border: 1px solid {accent}; background: {soft}; }}
    QPushButton:pressed {{ background: {mid}; padding-top: 9px; padding-bottom: 7px; }}
    QPushButton:checked {{ background: {mid}; border: 1px solid {accent}; }}
    QPushButton:disabled {{ color: {MUTED}; }}
    QPushButton#primary {{ background: {accent}; color: {on_accent}; font-weight: 700; border: none; }}
    QPushButton#danger {{ background: {DANGER}; color: white; font-weight: 700; border: none; }}
    QPushButton#modeOn {{ background: {accent}; color: {on_accent}; font-weight: 700; border: none; }}
    QFrame#banner {{ background: rgba(255,181,71,30); border: 1px solid {WARN}; border-radius: 10px; }}
    QTabWidget::pane {{ border: 1px solid {soft}; border-radius: 10px; background: {P.panel2}; }}
    QTabBar::tab {{ background: {PANEL_2}; color: {MUTED}; padding: 8px 14px; border-radius: 6px; margin: 2px; }}
    QTabBar::tab:selected {{ color: {on_accent}; background: {accent}; }}
    QCheckBox {{ background: transparent; spacing: 8px; }}
    QTableWidget {{ background: {PANEL_2}; gridline-color: {soft}; border: none; border-radius: 8px; }}
    QHeaderView::section {{ background: {BG}; color: {MUTED}; border: none; padding: 6px; }}
    QScrollBar:vertical {{ background: transparent; width: 8px; }}
    QScrollBar::handle:vertical {{ background: {mid}; border-radius: 4px; min-height: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
    QScrollBar::handle:vertical:hover {{ background: {accent}; }}
    QScrollArea, QWidget#chatInner, QWidget#chatRow {{ background: transparent; border: none; }}
    QFrame#bubbleAi {{ background: {PANEL_2}; border: 1px solid {soft}; border-radius: 14px;
                       border-top-left-radius: 4px; }}
    QFrame#bubbleMe {{ background: {mid}; border: 1px solid {accent}; border-radius: 14px;
                       border-top-right-radius: 4px; }}
    QFrame#card {{ background: {PANEL_2}; border: 1px solid {soft}; border-radius: 12px; }}
    QFrame#cardGood {{ background: rgba(61,220,151,{45 if P.light else 28}); border: 1px solid {OK}; border-radius: 12px; }}
    QFrame#cardBad {{ background: rgba(255,92,108,{45 if P.light else 28}); border: 1px solid {DANGER}; border-radius: 12px; }}
    QLabel#who {{ color: {strong}; font-size: 10px; font-weight: 700; letter-spacing: 1px; background: transparent; }}
    QLabel#msg {{ color: {TEXT}; font-size: 14px; background: transparent; }}
    QLabel#toolOk, QLabel#toolErr {{ font-family: Consolas, monospace; font-size: 11px; padding: 3px 10px;
                                     border-radius: 9px; background: {soft}; }}
    QLabel#toolOk {{ color: {good_text}; border: 1px solid rgba(61,220,151,110); }}
    QLabel#toolErr {{ color: {warn_text}; border: 1px solid rgba(255,181,71,130); }}
    QLabel#caption {{ color: {TEXT}; font-size: 13px; font-style: italic; background: transparent; }}
    QLabel#dayline {{ color: {MUTED}; font-size: 10px; font-weight: 700; letter-spacing: 2px; background: transparent; }}
    QLabel#good {{ color: {good_text}; font-weight: 700; background: transparent; }}
    QLabel#bad {{ color: {DANGER}; font-weight: 700; background: transparent; }}
    QPushButton#chip {{ background: transparent; border: 1px solid {soft}; border-radius: 13px;
                        padding: 4px 11px; font-size: 12px; color: {MUTED}; }}
    QPushButton#chip:hover {{ color: {TEXT}; border: 1px solid {accent}; background: {soft}; }}
    QPushButton#chip:checked {{ color: {on_accent}; background: {accent}; border: 1px solid {accent}; }}
    QPushButton#small {{ background: transparent; border: none; color: {MUTED}; padding: 2px 6px; }}
    QPushButton#small:hover {{ color: {strong}; }}
    QPushButton#option {{ text-align: left; padding: 10px 14px; font-size: 13px; }}
    QPushButton#optionGood {{ text-align: left; padding: 10px 14px; font-size: 13px; color: {TEXT};
                              background: rgba(61,220,151,70); border: 1px solid {OK}; }}
    QPushButton#optionBad {{ text-align: left; padding: 10px 14px; font-size: 13px; color: {TEXT};
                             background: rgba(255,92,108,70); border: 1px solid {DANGER}; }}
    QPushButton#jump {{ background: {accent}; color: {on_accent}; border: none; border-radius: 13px;
                        padding: 4px 12px; font-weight: 700; }}
    QProgressBar {{ background: {P.panel2}; border: 1px solid {soft}; border-radius: 6px; max-height: 12px;
                    text-align: center; color: {TEXT}; font-size: 9px; }}
    QProgressBar::chunk {{ background: {accent}; border-radius: 5px; }}
    """
