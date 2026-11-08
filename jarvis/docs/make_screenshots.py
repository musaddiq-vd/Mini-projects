"""Make the README screenshots from demo data (never your own data).

Run from the project folder:  .venv\\Scripts\\python docs\\make_screenshots.py
Everything happens in a temporary Jarvis home with a made-up user; nothing is sent anywhere.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
sys.path.insert(0, str(ROOT))
os.environ["JARVIS_HOME"] = tempfile.mkdtemp(prefix="jarvis-demo-")

from PyQt6.QtCore import QEventLoop, QPoint, Qt, QTimer  # noqa: E402
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPixmap  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

qt = QApplication(sys.argv)
qt.setFont(QFont("Segoe UI", 10))

from jarvis.achievements import Achievements  # noqa: E402
from jarvis.app import create_app  # noqa: E402

Achievements._on_event = lambda self, kind, data: None  # demo shots: no badge toasts or confetti
from jarvis.config import Settings, save_settings  # noqa: E402
from jarvis.ui import theme  # noqa: E402


def fresh_home() -> None:
    """Every scene gets its own empty demo home."""
    os.environ["JARVIS_HOME"] = tempfile.mkdtemp(prefix="jarvis-demo-")


def online(w) -> None:
    w.big_orb.boot = 1.0
    w._on_state("listening")
    w._set_net("online")


def wait(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def offscreen(widget) -> None:
    widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    widget.show()


def save(pix: QPixmap, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pix.save(str(OUT / name))
    print("saved", name, pix.width(), "x", pix.height())


def wallpaper(w: int, h: int) -> QPixmap:
    pix = QPixmap(w, h)
    p = QPainter(pix)
    g = QLinearGradient(0, 0, w, h)
    g.setColorAt(0, QColor("#1d3557"))
    g.setColorAt(0.55, QColor("#5b3f8c"))
    g.setColorAt(1, QColor("#e07a5f"))
    p.fillRect(pix.rect(), g)
    p.end()
    return pix


def demo_data(app) -> None:
    from jarvis.tools.goals import goals
    from jarvis.tools.habits import habits
    from jarvis.tools.money import expenses

    now, today = time.time(), date.today()
    monday = today - timedelta(days=today.weekday() + 7)  # last week: a full week of demo life
    week_end = monday + timedelta(days=6)
    app.state.set("reminders", [
        {"id": 1, "due": now + 25 * 60, "message": "Call the bank about the card", "repeat": ""},
        {"id": 2, "due": now + 3 * 3600, "message": "Team standup notes", "repeat": "weekdays"}])
    app.state.set("notes", [
        {"id": 1, "text": "Pay electricity bill", "todo": True, "done": False, "created": now - 9000,
         "due": (today + timedelta(days=1)).isoformat(), "priority": "high"},
        {"id": 2, "text": "Book train tickets", "todo": True, "done": False, "created": now - 8000,
         "due": today.isoformat()},
        {"id": 3, "text": "Buy milk and bread", "todo": True, "done": False, "created": now - 7000},
        {"id": 4, "text": "Send the project report", "todo": True, "done": True, "created": now - 90000,
         "done_at": now - 3600}])
    app.state.set("focus_log", [{"date": (monday + timedelta(days=i)).isoformat(), "minutes": m}
                                for i, m in enumerate([50, 75, 25, 100, 60, 30, 45])])
    items = []
    for i, (amt, cat) in enumerate([(120, "food"), (450, "travel"), (80, "food"), (1299, "shopping"),
                                    (60, "food"), (200, "bills")]):
        d = monday + timedelta(days=i)
        items.append({"id": i + 1, "amount": amt, "category": cat, "note": "", "date": d.isoformat()})
    items.append({"id": 99, "amount": 3100, "category": "food", "note": "",
                  "date": (monday - timedelta(days=5)).isoformat()})
    expenses().save({"items": items, "budgets": {"total": 15000}, "next_id": 100})
    log = {(monday + timedelta(days=i)).isoformat(): 1 for i in range((week_end - monday).days + 1)}
    habits().save({"habits": [
        {"id": 1, "name": "Gym", "target": 1, "days": "daily", "icon": "💪", "log": log,
         "created_day": (today - timedelta(days=30)).isoformat()},
        {"id": 2, "name": "Read 20 pages", "target": 1, "days": "daily", "icon": "📖",
         "log": {k: 1 for k in list(log)[::2]}, "created_day": (today - timedelta(days=30)).isoformat()},
        {"id": 3, "name": "Water", "target": 8, "days": "daily", "icon": "💧",
         "log": {k: 8 for k in log}, "created_day": (today - timedelta(days=30)).isoformat()}], "next_id": 4})
    noon = datetime.combine(monday, datetime.min.time()).replace(hour=12).timestamp()
    goals().save({"goals": [{"id": 1, "title": "Learn Python", "deadline": (today + timedelta(days=60)).isoformat(),
                             "why": "", "done": False, "created": noon,
                             "steps": [{"text": "Basics", "due": monday.isoformat(), "done": True, "done_at": noon},
                                       {"text": "Functions", "due": week_end.isoformat(), "done": True,
                                        "done_at": noon + 3 * 86400},
                                       {"text": "A small project", "due": (today + timedelta(days=9)).isoformat(),
                                        "done": False}]}], "next_id": 2})


def main_window(mode: str, name: str, chat: bool = True) -> None:
    from jarvis.ui.window import Bridge, MainWindow

    fresh_home()
    theme.apply("Jarvis Blue", mode, "#00d4ff")
    app = create_app(Settings(user_name="Alex", theme_mode=mode, sound_effects=False))
    demo_data(app)
    w = MainWindow(app, Bridge())
    w.resize(1280, 800)
    offscreen(w)
    wait(300)
    online(w)
    if not chat:  # the welcome card of an empty chat
        wait(900)
        save(w.grab(), name)
        w.reminders.close_all()
        w.close()
        return
    ev = app.events.emit
    ev("user", text="What's the weather like today?")
    ev("tool", name="get_weather", args={"city": "Pune"}, result={
        "place": "Pune, India", "temperature_c": 27, "feels_like_c": 29, "now": "partly cloudy",
        "today_max_c": 31, "today_min_c": 19, "rain_chance_percent": 10, "humidity_percent": 48, "wind_kmh": 9})
    ev("assistant", text="It's 27° and partly cloudy in Pune, Alex - a lovely day. Low chance of rain, so no "
                         "umbrella needed.")
    ev("user", text="Remind me in 25 minutes to call the bank about the card")
    ev("tool", name="set_reminder", args={"message": "Call the bank about the card", "minutes": 25},
       result={"result": "Reminder #1 set for 25 minutes from now."})
    ev("assistant", text="Done! I'll remind you at the right time.")
    wait(1800)
    w.big_orb.set_state("speaking")
    w.big_orb.set_level(0.55)
    wait(250)
    save(w.grab(), name)
    w.reminders.close_all()
    w.close()


def cards() -> None:
    from jarvis.ui.cards import fix_card, report_card

    theme.apply("Jarvis Blue", "dark", "#00d4ff")
    qt.setStyleSheet(theme.stylesheet("#00d4ff"))
    fresh_home()
    app = create_app(Settings(user_name="Alex"))
    demo_data(app)
    report = app.registry.call("weekly_report", {"which": "last"}, app.ctx)
    fix = {"app": "Windows Terminal", "problem": "TypeError: can't multiply sequence by non-int of type 'str'",
           "cause": "`input()` always returns text, so `price * quantity` multiplies two strings.",
           "steps": ["Convert `price` with `float()`.", "Convert `quantity` with `int()`.", "Run the script again."],
           "code": "price = float(input('Price: '))\nquantity = int(input('Quantity: '))\ntotal = price * quantity",
           "language": "python"}
    shots = []
    for card in (report_card(report), fix_card(fix)):
        card.setFixedWidth(520)
        offscreen(card)
        wait(150)
        shots.append(card.grab())
    gap = 24
    dpr = shots[0].devicePixelRatio()  # grabs are in device pixels on a scaled screen
    sizes = [(round(s.width() / dpr), round(s.height() / dpr)) for s in shots]
    pix = QPixmap(round((sum(w for w, _ in sizes) + gap * 3) * dpr), round((max(h for _, h in sizes) + gap * 2) * dpr))
    pix.setDevicePixelRatio(dpr)
    pix.fill(QColor(theme.P.bg))
    p = QPainter(pix)
    x = gap
    for s, (w, _) in zip(shots, sizes):
        p.drawPixmap(x, gap, s)
        x += w + gap
    p.end()
    save(pix, "cards.png")


def popups_widget_mini() -> None:
    from jarvis.ui.desktop_widget import DesktopWidget
    from jarvis.ui.mini import MiniWindow, next_up
    from jarvis.ui.reminder_popup import ReminderPopups

    fresh_home()
    theme.apply("Jarvis Blue", "dark", "#00d4ff")
    app = create_app(Settings(user_name="Alex"))
    demo_data(app)
    base = wallpaper(1180, 560)
    p = QPainter(base)
    # desktop widget, top-left
    d = DesktopWidget(app, "#00d4ff")
    d.orb.set_state("listening")
    d.refresh_info()
    offscreen(d)
    # mini mode with chat, middle
    mini = MiniWindow("#00d4ff", True, lambda: None, [], send=lambda t: None, upcoming=lambda: next_up(app))
    mini.chat_open = True
    mini._apply_open()
    for who, text in [("user", "Pause the music and open my notes"), ("assistant", "Paused. Your notes are open."),
                      ("user", "What's next today?"), ("assistant", "Standup notes at 3 PM, then the bank call.")]:
        mini.add_message(who, text)
    mini.orb.set_state("listening")
    mini.orb.boot = 1.0
    d.orb.boot = 1.0
    offscreen(mini)
    # reminder pop-ups, bottom-right
    rp = ReminderPopups(lambda: "#00d4ff", on_done=print, on_snooze=print, on_cancel=print, on_open=lambda: None)
    due = rp.show("due", 1, "Call the bank about the card", time.time())
    made = rp.show("set", 2, "Team standup notes", time.time() + 3 * 3600, repeat="weekdays")
    for wdg in (due, made):
        wdg.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    wait(1500)
    p.drawPixmap(28, 28, d.grab())
    p.drawPixmap(470, 0, mini.grab())
    p.drawPixmap(base.width() - made.width() - 24, base.height() - due.height() - made.height() - 34, made.grab())
    p.drawPixmap(base.width() - due.width() - 24, base.height() - due.height() - 24, due.grab())
    p.end()
    save(base, "desktop.png")
    rp.close_all()
    for wdg in (d, mini):
        wdg.close()


def power_up() -> None:
    from jarvis.ui.orb import Orb

    o = Orb("#00d4ff")
    o.resize(300, 340)
    o.boot = 1.0
    o.set_state("listening")
    offscreen(o)
    wait(200)
    o.power_up()
    frames = []
    for t in (0.2, 0.36, 0.55, 0.85):
        o.power["t"] = t
        o._last = time.monotonic()
        pm = QPixmap(o.size())
        pm.fill(QColor("#0a0f14"))
        p = QPainter(pm)
        o.render(p, QPoint(0, 0))
        p.end()
        frames.append(pm)
    pix = QPixmap(300 * 4, 340)
    pix.fill(QColor("#0a0f14"))
    p = QPainter(pix)
    for i, f in enumerate(frames):
        p.drawPixmap(i * 300, 0, f)
    p.end()
    save(pix, "power-up.png")
    o.close()


def dashboard_and_planner() -> None:
    from jarvis.ui.window import Bridge, MainWindow

    fresh_home()
    theme.apply("Jarvis Blue", "dark", "#00d4ff")
    app = create_app(Settings(user_name="Alex", api_key="demo", reply_language="hinglish"))
    save_settings(app.settings)
    demo_data(app)
    w = MainWindow(app, Bridge())
    w.resize(1280, 800)
    offscreen(w)
    online(w)
    w.toggle_dashboard("week")
    wait(900)
    save(w.grab(), "dashboard.png")
    w.toggle_dashboard()
    w.toggle_planner()
    wait(600)
    save(w.grab(), "planner.png")
    w.reminders.close_all()
    w.close()


if __name__ == "__main__":
    main_window("dark", "main-dark.png")
    main_window("light", "welcome-light.png", chat=False)
    cards()
    popups_widget_mini()
    power_up()
    dashboard_and_planner()
    print("done:", OUT)
