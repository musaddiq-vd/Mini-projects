"""Rich cards in the chat for some results: weather, and web / news links you can click."""

from __future__ import annotations

import html
from urllib.parse import urlparse

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout

from jarvis.ui import theme

WEATHER_ICONS = [("thunder", "⛈"), ("snow", "❄"), ("rain", "🌧"), ("drizzle", "🌦"), ("shower", "🌧"),
                 ("fog", "🌫"), ("overcast", "☁"), ("partly", "⛅"), ("mainly clear", "🌤"), ("clear", "☀")]


def weather_icon(condition: str) -> str:
    c = (condition or "").lower()
    return next((icon for key, icon in WEATHER_ICONS if key in c), "🌡")


def _num(value) -> str:
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return "–"


def weather_card(r: dict) -> QFrame | None:
    if not isinstance(r, dict) or "temperature_c" not in r:
        return None
    card = QFrame(objectName="card")
    card.setMaximumWidth(460)
    v = QVBoxLayout(card)
    v.setContentsMargins(16, 12, 16, 14)
    v.setSpacing(6)
    place = QLabel(f"📍 {r.get('place', '')}", objectName="muted")
    v.addWidget(place)
    top = QHBoxLayout()
    icon = QLabel(weather_icon(r.get("now", "")))
    icon.setStyleSheet("font-size: 44px; background: transparent;")
    top.addWidget(icon)
    temp = QLabel(f"{_num(r['temperature_c'])}°")
    temp.setStyleSheet(f"font-size: 40px; font-weight: 700; color: {theme.readable(theme.ACCENT)}; "
                       "background: transparent;")
    top.addWidget(temp)
    side = QVBoxLayout()
    side.addWidget(QLabel(str(r.get("now", "")).capitalize(), objectName="h2"))
    side.addWidget(QLabel(f"Feels like {_num(r.get('feels_like_c'))}°", objectName="muted"))
    top.addLayout(side)
    top.addStretch(1)
    v.addLayout(top)
    grid = QGridLayout()
    grid.setHorizontalSpacing(16)
    stats = [("⬆", f"{_num(r.get('today_max_c'))}°", "High"), ("⬇", f"{_num(r.get('today_min_c'))}°", "Low"),
             ("☔", f"{_num(r.get('rain_chance_percent'))}%", "Rain"), ("💧", f"{_num(r.get('humidity_percent'))}%", "Humidity"),
             ("🌬", f"{_num(r.get('wind_kmh'))} km/h", "Wind")]
    for i, (ic, value, name) in enumerate(stats):
        cell = QLabel(f"{ic} <b>{value}</b><br><span style='color:{theme.MUTED}; font-size:10px;'>{name}</span>")
        cell.setTextFormat(Qt.TextFormat.RichText)
        cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(cell, 0, i)
    v.addLayout(grid)
    return card


def _safe_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    return url if parsed.scheme in ("http", "https") and parsed.netloc else None


def links_card(r: dict, limit: int = 5) -> QFrame | None:
    results = r.get("results") if isinstance(r, dict) else None
    if not results:
        return None
    news = r.get("mode") == "news"
    card = QFrame(objectName="card")
    card.setMaximumWidth(580)
    card.setMinimumWidth(420)
    v = QVBoxLayout(card)
    v.setContentsMargins(14, 10, 14, 12)
    v.setSpacing(8)
    head = f"{'📰 News' if news else '🔎 Web'}  ·  {html.escape(str(r.get('query', '')))}"
    v.addWidget(QLabel(head, objectName="section"))
    for item in results[:limit]:
        url = _safe_url(item.get("url"))
        title = html.escape(str(item.get("title") or "Untitled"))
        domain = urlparse(url).netloc.removeprefix("www.") if url else ""
        meta = " · ".join(x for x in (html.escape(str(item.get("source") or "")) or domain,
                                      html.escape(str(item.get("date") or ""))[:10]) if x)
        snippet = html.escape(str(item.get("snippet") or ""))[:140]
        link_color = theme.readable(theme.ACCENT)
        title_html = f'<a href="{html.escape(url)}" style="color:{link_color}; text-decoration:none;">{title}</a>' \
            if url else f"<span>{title}</span>"
        lab = QLabel(f"{title_html}<br><span style='color:{theme.MUTED}; font-size:10px;'>{meta}</span>"
                     + (f"<br><span style='font-size:11px;'>{snippet}…</span>" if snippet else ""))
        lab.setTextFormat(Qt.TextFormat.RichText)
        lab.setWordWrap(True)
        lab.setOpenExternalLinks(False)
        lab.linkActivated.connect(lambda link: QDesktopServices.openUrl(QUrl(link)))
        if url:
            lab.setToolTip(url)
        v.addWidget(lab)
    return card


def _money(value: float, currency: str) -> str:
    return f"{currency}{value:,.0f}" if value >= 100 else f"{currency}{value:,.2f}".rstrip("0").rstrip(".")


def report_card(r: dict) -> QFrame | None:
    """The weekly report: six stat tiles, top spending, wins and goals."""
    if not isinstance(r, dict) or "week" not in r or "spent" not in r:
        return None
    cur = r.get("currency", "₹")
    card = QFrame(objectName="card")
    card.setMaximumWidth(520)
    v = QVBoxLayout(card)
    v.setContentsMargins(16, 12, 16, 14)
    v.setSpacing(8)
    title = "Your week" if r.get("which") != "last" else "Last week"
    v.addWidget(QLabel(f"📊  {title}  ·  {html.escape(r['week'])}", objectName="h2"))
    change = r.get("spent_change_percent")
    trend = "" if change is None else (f"▲ {change}%" if change > 0 else f"▼ {abs(change)}%" if change < 0 else "same")
    trend_color = theme.DANGER if change and change > 0 else theme.OK
    tiles = [
        ("💰", _money(r["spent"], cur), "spent" + (f" <span style='color:{trend_color}'>{trend}</span>" if trend else "")),
        ("✅", f"{r['habits_percent']}%" if r.get("habits_percent") is not None else "–", "habits done"),
        ("🎯", str(r.get("goal_steps_done", 0)), "goal steps"),
        ("📝", str(r.get("tasks_done", 0)), "tasks done"),
        ("⏱", r.get("focus") or "–", "focus time"),
        ("📅", "–" if r.get("meetings_next_week") is None else str(r["meetings_next_week"]), "meetings ahead"),
    ]
    grid = QGridLayout()
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(8)
    accent = theme.readable(theme.ACCENT)
    for i, (icon, value, name) in enumerate(tiles):
        tile = QLabel(f"<div style='font-size:15px'>{icon}</div>"
                      f"<div style='font-size:18px; font-weight:700; color:{accent}'>{html.escape(value)}</div>"
                      f"<div style='color:{theme.MUTED}; font-size:10px'>{name}</div>")
        tile.setTextFormat(Qt.TextFormat.RichText)
        tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tile.setStyleSheet(f"background: {theme.P.panel2}; border-radius: 10px; padding: 6px;")
        grid.addWidget(tile, i // 3, i % 3)
    v.addLayout(grid)
    lines = []
    if r.get("top_categories"):
        lines.append("Top spend: " + " · ".join(f"{html.escape(str(c)).title()} {_money(a, cur)}"
                                                 for c, a in r["top_categories"]))
    if r.get("wins"):
        lines.append("🏆 " + html.escape(", ".join(r["wins"])))
    for g in (r.get("goals") or [])[:2]:
        lines.append(f"🎯 {html.escape(g['title'])} — {g['percent']}%")
    for line in lines:
        lb = QLabel(line, objectName="muted")
        lb.setTextFormat(Qt.TextFormat.RichText)
        lb.setWordWrap(True)
        v.addWidget(lb)
    return card


def _inline_code(text: str) -> str:
    """Escape text and show `code` in a code font (models like backticks)."""
    import re

    accent = theme.readable(theme.ACCENT)
    return re.sub(r"`([^`]+)`", lambda m: f"<span style='font-family: Consolas, monospace; color:{accent}'>"
                  f"{m.group(1)}</span>", html.escape(text))


def fix_card(r: dict) -> QFrame | None:
    """The fix for an error on screen: problem, cause, steps and code with a Copy button."""
    if not isinstance(r, dict) or not r.get("problem"):
        return None
    from PyQt6.QtWidgets import QApplication, QPushButton

    card = QFrame(objectName="card")
    card.setMaximumWidth(560)
    v = QVBoxLayout(card)
    v.setContentsMargins(16, 12, 16, 14)
    v.setSpacing(6)
    head = QLabel("🛠  Fix" + (f"  ·  {html.escape(r['app'])}" if r.get("app") else ""), objectName="h2")
    v.addWidget(head)
    problem = QLabel(f"<b style='color:{theme.DANGER}'>{html.escape(r['problem'])}</b>")
    problem.setTextFormat(Qt.TextFormat.RichText)
    problem.setWordWrap(True)
    v.addWidget(problem)
    if r.get("cause"):
        cause = QLabel(_inline_code(r["cause"]), objectName="muted")
        cause.setTextFormat(Qt.TextFormat.RichText)
        cause.setWordWrap(True)
        v.addWidget(cause)
    for i, step in enumerate(r.get("steps") or [], 1):
        lb = QLabel(f"<b style='color:{theme.readable(theme.ACCENT)}'>{i}.</b> {_inline_code(step)}")
        lb.setTextFormat(Qt.TextFormat.RichText)
        lb.setWordWrap(True)
        v.addWidget(lb)
    code = r.get("code") or ""
    if code:
        box = QLabel(f"<pre style='margin:0'>{html.escape(code)}</pre>")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.setWordWrap(True)
        box.setStyleSheet(f"background:{theme.P.bg}; border:1px solid {theme.MUTED}; border-radius:8px; "
                          "padding:8px; font-family: Consolas, 'Cascadia Mono', monospace; font-size: 12px;")
        v.addWidget(box)
        row = QHBoxLayout()
        lang = QLabel(html.escape(r.get("language") or ""), objectName="muted")
        row.addWidget(lang)
        row.addStretch(1)
        copy = QPushButton("📋  Copy code", objectName="chip")
        copy.setCursor(Qt.CursorShape.PointingHandCursor)

        def do_copy():
            QApplication.clipboard().setText(code)
            copy.setText("✓  Copied")

        copy.clicked.connect(do_copy)
        row.addWidget(copy)
        v.addLayout(row)
    return card
