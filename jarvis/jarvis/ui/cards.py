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
