"""Rich cards, the evening recap and health nudges."""

import sys
import time
from datetime import datetime

import pytest


@pytest.fixture
def qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


# -- cards ------------------------------------------------------------------------------------
@pytest.mark.parametrize("cond, icon", [("clear sky", "☀"), ("partly cloudy", "⛅"), ("light rain", "🌧"),
                                         ("thunderstorm with hail", "⛈"), ("fog", "🌫"), ("???", "🌡")])
def test_weather_icons(cond, icon):
    from jarvis.ui.cards import weather_icon

    assert weather_icon(cond) == icon


def test_weather_card(qt):
    from PyQt6.QtWidgets import QLabel

    from jarvis.ui.cards import weather_card

    card = weather_card({"place": "Pune, India", "now": "clear sky", "temperature_c": 29.6, "feels_like_c": 31,
                         "humidity_percent": 40, "wind_kmh": 9, "today_max_c": 31, "today_min_c": 19,
                         "rain_chance_percent": 0})
    text = " ".join(lbl.text() for lbl in card.findChildren(QLabel))
    assert "30°" in text and "Pune" in text and "☀" in text
    assert weather_card({"error": "x"}) is None


def test_links_card_only_opens_web_links(qt):
    from PyQt6.QtWidgets import QLabel

    from jarvis.ui.cards import links_card

    card = links_card({"mode": "news", "query": "x", "results": [
        {"title": "Good <b>news</b>", "url": "https://example.com/a", "source": "Ex"},
        {"title": "Sneaky", "url": "javascript:alert(1)"},
        {"title": "Local file", "url": "file:///C:/Windows/win.ini"}]})
    html = " ".join(lbl.text() for lbl in card.findChildren(QLabel))
    assert 'href="https://example.com/a"' in html and "&lt;b&gt;news" in html
    assert "javascript:" not in html and "file:///" not in html
    assert links_card({"results": []}) is None


def test_window_shows_cards_instead_of_chips(qt, isolated_home):
    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    app = create_app(Settings())
    w = MainWindow(app, Bridge())
    app.events.emit("tool", name="get_weather", args={"city": "Pune"},
                    result={"place": "Pune", "now": "clear sky", "temperature_c": 25})
    qt.processEvents()
    text = w.chat.toPlainText()
    assert "25°" in text and "Get weather" not in text


# -- recap ------------------------------------------------------------------------------------
def test_daily_recap_collects_the_day(registry, ctx, isolated_home):
    from jarvis.config import home_dir
    from jarvis.history import ChatHistory
    from jarvis.state import StateStore

    ctx.state = StateStore(isolated_home / "state.json")
    ChatHistory(home_dir() / "history.jsonl").add("user", "plan my trip to Goa")
    registry.call("add_note", {"text": "book tickets"}, ctx)
    registry.call("add_note", {"text": "pack bags"}, ctx)
    registry.call("complete_note", {"item": "tickets"}, ctx)
    out = registry.call("daily_recap", {}, ctx)
    assert out["things_you_asked"] == ["plan my trip to Goa"]
    assert out["todos_done_today"] == ["book tickets"] and out["todos_still_open"] == ["pack bags"]
    assert "tomorrow" in out["how_to_answer"] and "language_practice" not in out


@pytest.fixture
def bg(isolated_home):
    from jarvis.app import create_app
    from jarvis.background import BackgroundServices
    from jarvis.config import Settings

    app = create_app(Settings())
    said = []
    b = BackgroundServices(app, said.append)
    b.said = said
    return b, app


def test_recap_only_in_the_evening_once_and_if_you_talked(bg):
    b, app = bg
    evening = datetime.now().replace(hour=21, minute=30)
    assert not b.recap_due(evening)  # no conversation today
    app.events.emit("user", text="hello")
    assert b.recap_due(evening)
    assert not b.recap_due(evening.replace(hour=15))
    app.state.set("last_recap", evening.date().isoformat())
    assert not b.recap_due(evening)  # once a day
    app.state.set("last_recap", None)
    b.sleeping = True
    assert not b.recap_due(evening)  # waits until Jarvis is awake
    b.sleeping = False
    app.settings.evening_recap = False
    assert not b.recap_due(evening)


def test_health_nudges(bg):
    b, app = bg
    app.settings.eye_break_minutes, app.settings.water_minutes = 50, 90
    t = 1_000_000.0
    got = []
    for _ in range(int(102 * 60 / b.HEALTH_TICK)):
        t += b.HEALTH_TICK
        kind = b.health_step(idle=5, now=t, hour=14)
        if kind:
            got.append((kind, round((t - 1_000_000) / 60)))
    assert ("eyes", 50) in got and any(k == "water" for k, _ in got) and ("eyes", 100) in got


def test_stepping_away_counts_as_a_break_and_quiet_hours(bg):
    b, app = bg
    t = 0.0
    for _ in range(int(40 * 60 / b.HEALTH_TICK)):
        t += b.HEALTH_TICK
        b.health_step(idle=5, now=t, hour=14)
    b.health_step(idle=600, now=t + 30, hour=14)  # away for 10 minutes
    t += 30
    kinds = []
    for _ in range(int(30 * 60 / b.HEALTH_TICK)):
        t += b.HEALTH_TICK
        kinds.append(b.health_step(idle=5, now=t, hour=14))
    assert "eyes" not in kinds  # the clock restarted after the break
    assert b.health_step(idle=5, now=t + 99999, hour=2) is None  # night: quiet
    app.settings.health_nudges = False
    assert b.health_step(idle=5, now=t + 99999, hour=14) is None


def test_window_shows_nudges(qt, isolated_home):
    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    app = create_app(Settings())
    w = MainWindow(app, Bridge())
    w.show()
    app.events.emit("nudge", kind="water", text="💧 Time for a glass of water.")
    qt.processEvents()
    assert any("water" in t.findChildren(type(w.title))[1].text() for t in w.toasts.toasts)
    assert w.big_orb.cute_kind == "happy"
    assert time.monotonic() < w.big_orb.cute_until


def test_events_accept_a_kind_field():
    from jarvis.events import Events

    ev = Events()
    seen = []
    ev.subscribe(lambda k, d: seen.append((k, d)))
    ev.emit("nudge", kind="water", text="x")  # used to raise TypeError
    assert seen == [("nudge", {"kind": "water", "text": "x"})]
