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


def digest_card(r: dict) -> QFrame | None:
    """The daily digest: weather, meetings, to-dos, reminders, habits and money for today."""
    if not isinstance(r, dict) or "digest" not in r:
        return None
    card = QFrame(objectName="card")
    card.setMaximumWidth(520)
    v = QVBoxLayout(card)
    v.setContentsMargins(16, 12, 16, 14)
    v.setSpacing(6)
    v.addWidget(QLabel(f"🌅  {html.escape(r.get('greeting', 'Your day'))}  ·  {html.escape(r['digest'])}",
                       objectName="h2"))
    accent = theme.readable(theme.ACCENT)

    def section(icon: str, title: str, lines: list[str]) -> None:
        if not lines:
            return
        body = "<br>".join(html.escape(str(x)) for x in lines)
        lb = QLabel(f"<b style='color:{accent}'>{icon} {title}</b><br><span>{body}</span>")
        lb.setTextFormat(Qt.TextFormat.RichText)
        lb.setWordWrap(True)
        lb.setStyleSheet(f"background: {theme.P.panel2}; border-radius: 10px; padding: 6px 10px;")
        v.addWidget(lb)

    w = r.get("weather")
    if w:
        section(weather_icon(w.get("now", "")), "Weather",
                [f"{w.get('place', '')}: {w.get('now', '')}, {_num(w.get('temperature_c'))}°C "
                 f"(max {_num(w.get('today_max_c'))}° / min {_num(w.get('today_min_c'))}°)"
                 + (f", {w['rain_chance_percent']}% rain" if w.get("rain_chance_percent") else "")])
    section("📅", "Meetings", [f"{m['time']}  {m['title']}" for m in r.get("meetings") or []])
    t = r.get("todos") or {}
    if t.get("first"):
        extra = []
        if t.get("overdue"):
            extra.append(f"{t['overdue']} overdue")
        if t.get("due_today"):
            extra.append(f"{t['due_today']} due today")
        section("📝", f"Do first ({t.get('open', 0)} open{', ' + ', '.join(extra) if extra else ''})",
                [f"{i + 1}. {x}" for i, x in enumerate(t["first"])])
    section("⏰", "Reminders", r.get("reminders") or [])
    section("🎂", "Birthdays", r.get("birthdays") or [])
    section("✅", "Habits", [f"{h['habit']}  {h['today']}  ·  🔥 {h['streak']}" for h in r.get("habits") or []])
    money_lines = [x for x in (f"Spent today: {r['spent_today']}" if r.get("spent_today") else "",
                               f"This month: {r['budget']}" if r.get("budget") else "") if x]
    section("💰", "Money", money_lines)
    if r.get("result"):
        lb = QLabel(html.escape(r["result"]), objectName="muted")
        lb.setWordWrap(True)
        v.addWidget(lb)
    return card


def suggestion_card(data: dict, answer) -> QFrame:
    """A proactive suggestion with Accept / Dismiss / Not useful. `answer(outcome)` gets the click;
    `card.mark(outcome)` shows the answer (also when it was given by voice)."""
    from PyQt6.QtWidgets import QPushButton

    card = QFrame(objectName="card")
    card.setMaximumWidth(520)
    v = QVBoxLayout(card)
    v.setContentsMargins(14, 10, 14, 12)
    v.setSpacing(6)
    text = QLabel(f"{html.escape(data.get('icon', '💡'))}  {html.escape(data.get('text', ''))}")
    text.setTextFormat(Qt.TextFormat.RichText)
    text.setWordWrap(True)
    v.addWidget(text)
    row = QHBoxLayout()
    buttons = []
    for label, outcome in (("✓  Accept", "accepted"), ("Dismiss", "dismissed"), ("Not useful", "not_useful")):
        b = QPushButton(label, objectName="chip")
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.clicked.connect(lambda _, o=outcome: (card.mark(o), answer(o)))
        row.addWidget(b)
        buttons.append(b)
    row.addStretch(1)
    v.addLayout(row)
    status = QLabel("", objectName="muted")
    status.hide()
    v.addWidget(status)

    def mark(outcome: str) -> None:
        for b in buttons:
            b.setEnabled(False)
        status.setText({"accepted": "✓ Accepted", "dismissed": "Dismissed",
                        "not_useful": "Muted - you won't see this kind again"}.get(outcome, outcome))
        status.show()

    card.mark = mark
    return card


def organize_card(r: dict) -> QFrame | None:
    """The organize plan (or result): "N files -> folders"."""
    if not isinstance(r, dict) or not r.get("folders"):
        return None
    card = QFrame(objectName="card")
    card.setMaximumWidth(520)
    v = QVBoxLayout(card)
    v.setContentsMargins(16, 12, 16, 14)
    v.setSpacing(4)
    done = "moved" in r
    count = r.get("moved", r.get("files", 0))
    name = html.escape(str(r.get("folder", "")).replace("\\", "/").rstrip("/").split("/")[-1])
    v.addWidget(QLabel(f"🗂  {'Organized' if done else 'Plan for'} {name}  ·  {count} files → "
                       f"{len(r['folders'])} folders", objectName="h2"))
    accent = theme.readable(theme.ACCENT)
    for dest, n in list(r["folders"].items())[:10]:
        lb = QLabel(f"<b style='color:{accent}'>{n}</b>  →  {html.escape(dest)}")
        lb.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(lb)
    if len(r["folders"]) > 10:
        v.addWidget(QLabel(f"… and {len(r['folders']) - 10} more folders", objectName="muted"))
    skipped = sum((r.get("skipped") or {}).values())
    foot = ("Say “undo” to put everything back." if done else "Nothing has moved yet - say “yes, do it”.")
    v.addWidget(QLabel((f"{skipped} skipped (in use, hidden or unknown).  " if skipped else "") + foot,
                       objectName="muted"))
    return card
