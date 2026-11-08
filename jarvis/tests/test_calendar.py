"""Phase C: calendar sync from private iCal links."""

import json
import sys
import time
from datetime import date, datetime, timedelta

import pytest

from jarvis.ics import expand, parse_duration, unfold

URL = "https://calendar.google.com/calendar/ical/me%40gmail.com/private-abc123secret/basic.ics"


def cal(*events: str, name: str = "Work") -> str:
    body = "".join(f"BEGIN:VEVENT\r\n{e.strip()}\r\nEND:VEVENT\r\n" for e in events)
    return f"BEGIN:VCALENDAR\r\nVERSION:2.0\r\nX-WR-CALNAME:{name}\r\n{body}END:VCALENDAR\r\n"


def stamp(d: datetime) -> str:
    return d.strftime("%Y%m%dT%H%M%S")


def titles(occ):
    return [(o.start.strftime("%d %H:%M"), o.title) for o in occ]


# -- parser -----------------------------------------------------------------------------------
def test_weekly_with_skipped_and_moved_instances():
    text = cal("""UID:s
SUMMARY:Standup
DTSTART;TZID=India Standard Time:20261102T100000
DTEND;TZID=India Standard Time:20261102T101500
RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR
EXDATE;TZID=India Standard Time:20261104T100000""", """UID:s
RECURRENCE-ID:20261106T043000Z
SUMMARY:Standup (late)
DTSTART;TZID=Asia/Kolkata:20261106T120000
DTEND;TZID=Asia/Kolkata:20261106T121500""")
    occ = expand(text, datetime(2026, 11, 2).astimezone(), datetime(2026, 11, 10).astimezone())
    from zoneinfo import ZoneInfo

    ist = [(o.start.astimezone(ZoneInfo("Asia/Kolkata")).strftime("%d %H:%M"), o.title) for o in occ]
    assert ist == [("02 10:00", "Standup"), ("06 12:00", "Standup (late)"), ("09 10:00", "Standup")]
    assert occ[0].end - occ[0].start == timedelta(minutes=15)


def test_cancelled_single_instance_and_cancelled_event():
    text = cal("""UID:d
SUMMARY:Gym
DTSTART:20261101T070000
RRULE:FREQ=DAILY;COUNT=4""", """UID:d
RECURRENCE-ID:20261102T070000
STATUS:CANCELLED
DTSTART:20261102T070000""", """UID:x
SUMMARY:Gone
STATUS:CANCELLED
DTSTART:20261101T090000""")
    occ = expand(text, datetime(2026, 11, 1), datetime(2026, 11, 10))
    assert titles(occ) == [("01 07:00", "Gym"), ("03 07:00", "Gym"), ("04 07:00", "Gym")]


def test_monthly_last_friday_count_and_31st():
    text = cal("""UID:r
SUMMARY:Review
DTSTART:20260130T150000
RRULE:FREQ=MONTHLY;BYDAY=-1FR;COUNT=3""", """UID:p
SUMMARY:Pay rent
DTSTART:20260131T090000
RRULE:FREQ=MONTHLY""")
    occ = expand(text, datetime(2026, 1, 1), datetime(2026, 5, 1))
    review = [o.start.date() for o in occ if o.title == "Review"]
    assert review == [date(2026, 1, 30), date(2026, 2, 27), date(2026, 3, 27)]
    rent = [o.start.date() for o in occ if o.title == "Pay rent"]
    assert rent == [date(2026, 1, 31), date(2026, 3, 31)]  # months without a 31st are skipped


def test_yearly_all_day_birthday_and_leap_day():
    text = cal("""UID:b
SUMMARY:Mom's birthday
DTSTART;VALUE=DATE:19700315
RRULE:FREQ=YEARLY""", """UID:l
SUMMARY:Leap
DTSTART;VALUE=DATE:20240229
RRULE:FREQ=YEARLY""")
    occ = expand(text, datetime(2026, 1, 1), datetime(2029, 1, 1))
    assert [o.start.year for o in occ if o.title == "Mom's birthday"] == [2026, 2027, 2028]
    assert [o.start.date() for o in occ if o.title == "Leap"] == [date(2028, 2, 29)]
    assert all(o.all_day and o.end - o.start == timedelta(days=1) for o in occ)


def test_multi_day_all_day_event_overlaps_window():
    text = cal("""UID:t
SUMMARY:Goa trip
DTSTART;VALUE=DATE:20261105
DTEND;VALUE=DATE:20261108""")
    occ = expand(text, datetime(2026, 11, 7), datetime(2026, 11, 8))
    assert titles(occ) == [("05 00:00", "Goa trip")] and occ[0].all_day


def test_until_in_utc_interval_and_duration():
    text = cal("""UID:u
SUMMARY:Class
DTSTART:20261101T180000
DURATION:PT1H30M
RRULE:FREQ=DAILY;INTERVAL=2;UNTIL=20261107T235959Z""")
    occ = expand(text, datetime(2026, 11, 1), datetime(2026, 12, 1))
    assert [o.start.day for o in occ] == [1, 3, 5, 7]
    assert occ[0].end - occ[0].start == timedelta(hours=1, minutes=30)
    assert parse_duration("P1W2D") == timedelta(days=9) and parse_duration("junk") is None


def test_folded_lines_escapes_links_and_garbage():
    text = ("BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nUID:f\r\nSUMMARY:Lunch\\, with Riya\\; and team\r\n"
            "DESCRIPTION:Join here: https://us02web.zoom.us/j/8812345\r\n 6789?pwd=abc\r\n"
            "this line is broken\r\nLOCATION:Cafe\\nBlue\r\nDTSTART:20261110T130000\r\n"
            "DTEND:20261110T140000\r\nEND:VEVENT\r\n"
            "BEGIN:VEVENT\r\nUID:bad\r\nSUMMARY:No start\r\nEND:VEVENT\r\n"
            "BEGIN:VEVENT\r\nUID:bad2\r\nSUMMARY:Bad date\r\nDTSTART:2026-11-10\r\nEND:VEVENT\r\nEND:VCALENDAR")
    occ = expand(text, datetime(2026, 11, 10), datetime(2026, 11, 11))
    assert len(occ) == 1 and occ[0].title == "Lunch, with Riya; and team"
    assert occ[0].link == "https://us02web.zoom.us/j/88123456789?pwd=abc"
    assert occ[0].location == "Cafe\nBlue"
    assert unfold("A\r\n B\r\n\tC\r\n") == ["ABC"]


def test_unknown_zone_uses_the_files_offset_and_utc_times():
    text = ("BEGIN:VCALENDAR\r\nBEGIN:VTIMEZONE\r\nTZID:Custom Zone\r\nBEGIN:STANDARD\r\n"
            "DTSTART:19700101T000000\r\nTZOFFSETFROM:+0530\r\nTZOFFSETTO:+0530\r\nEND:STANDARD\r\nEND:VTIMEZONE\r\n"
            "BEGIN:VEVENT\r\nUID:z\r\nSUMMARY:Zoned\r\nDTSTART;TZID=Custom Zone:20261110T100000\r\nEND:VEVENT\r\n"
            "BEGIN:VEVENT\r\nUID:u\r\nSUMMARY:Utc\r\nDTSTART:20261110T043000Z\r\nEND:VEVENT\r\nEND:VCALENDAR")
    occ = expand(text, datetime(2026, 11, 9), datetime(2026, 11, 12))
    assert len(occ) == 2 and occ[0].start == occ[1].start  # 10:00 +05:30 == 04:30 UTC


def test_huge_daily_rule_is_bounded():
    text = cal("""UID:h
SUMMARY:Forever
DTSTART:20000101T080000
RRULE:FREQ=DAILY""")
    t = time.monotonic()
    occ = expand(text, datetime(2026, 11, 1), datetime(2026, 11, 3))
    assert [o.start.day for o in occ] == [1, 2] and time.monotonic() - t < 2


# -- sync -------------------------------------------------------------------------------------
class Resp:
    def __init__(self, text="", status=200):
        self.status_code, self._b, self.encoding = status, text.encode("utf-8"), "utf-8"

    def iter_content(self, n):
        for i in range(0, len(self._b), n):
            yield self._b[i:i + n]


def soon_cal(minutes=8, title="Design review"):
    s = datetime.now().replace(second=0, microsecond=0) + timedelta(minutes=minutes)
    return cal(f"""UID:m1
SUMMARY:{title}
DTSTART:{stamp(s)}
DTEND:{stamp(s + timedelta(minutes=30))}
LOCATION:https://meet.google.com/abc-defg-hij""", """UID:a1
SUMMARY:Holiday
DTSTART;VALUE=DATE:""" + date.today().strftime("%Y%m%d"))


@pytest.fixture
def settings(isolated_home):
    from jarvis.config import Settings

    from jarvis.tools import calendar_sync

    calendar_sync._mem.update(mtime=None, data=None)
    return Settings(calendar_urls=[URL])


def test_sync_stores_events_and_keeps_last_copy_on_failure(settings):
    from jarvis.tools import calendar_sync as cal_

    out = cal_.sync(settings, get=lambda *a, **k: Resp(soon_cal()))
    assert out == {"calendars": 1, "events": 2, "errors": []}
    assert {e["title"] for e in cal_.events(settings)} == {"Design review", "Holiday"}
    assert not cal_.due(settings)
    out = cal_.sync(settings, get=lambda *a, **k: Resp("Not found", 404))
    assert "doesn't work any more" in out["errors"][0] and "private-abc123" not in json.dumps(out)
    assert len(cal_.events(settings)) == 2  # last good copy kept
    raw = cal_.data_file().path.read_text(encoding="utf-8")
    assert "abc123secret" not in raw and "gmail" not in raw  # the link itself is never stored here


@pytest.mark.parametrize("make, message", [
    (lambda *a, **k: Resp("<html>Sign in</html>"), "didn't return a calendar"),
    (lambda *a, **k: (_ for _ in ()).throw(ConnectionError(URL)), "no internet"),
    (lambda *a, **k: Resp("x", 500), "error 500"),
])
def test_fetch_errors_never_contain_the_link(make, message):
    from jarvis.tools.calendar_sync import fetch

    with pytest.raises(ValueError) as err:
        fetch(URL, make)
    assert message in str(err.value) and "abc123" not in str(err.value)


def test_link_cleanup_and_checks():
    from types import SimpleNamespace

    from jarvis.tools.calendar_sync import check_link, links

    s = SimpleNamespace(calendar_urls=["webcal://p01.icloud.com/x.ics  https://a.com/b.ics, https://a.com/b.ics", ""])
    assert links(s) == ["https://p01.icloud.com/x.ics", "https://a.com/b.ics"]
    assert "normal Google Calendar page" in check_link("https://calendar.google.com/calendar/u/0/r")
    assert "isn't a web link" in check_link("C:/cal.ics")
    assert check_link(URL) is None


def test_too_big_calendar(monkeypatch):
    from jarvis.tools import calendar_sync as cal_

    monkeypatch.setattr(cal_, "MAX_BYTES", 100)
    with pytest.raises(ValueError, match="too big"):
        cal_.fetch(URL, lambda *a, **k: Resp("BEGIN:VCALENDAR\n" + "X" * 500))


def test_two_calendars_same_meeting_once_and_removed_links_clear(settings):
    from jarvis.tools import calendar_sync as cal_

    settings.calendar_urls = [URL, "https://outlook.office365.com/owa/calendar/x/reachcalendar.ics"]
    cal_.sync(settings, get=lambda *a, **k: Resp(soon_cal()))
    assert len(cal_.events(settings)) == 2
    settings.calendar_urls = []
    assert cal_.events(settings) == [] and cal_.due(settings)
    cal_.sync(settings)
    assert not cal_.data_file().exists()


def test_links_are_encrypted_on_disk(isolated_home):
    from jarvis import secure
    from jarvis.config import Settings, load_settings, save_settings, settings_path

    save_settings(Settings(calendar_urls=[URL]))
    raw = settings_path().read_text(encoding="utf-8")
    if secure.available():
        assert "abc123secret" not in raw
    assert load_settings().calendar_urls == [URL]
    settings_path().write_text(json.dumps({"calendar_urls": "oops"}), encoding="utf-8")
    assert load_settings().calendar_urls == []


# -- tools ------------------------------------------------------------------------------------
def tomorrow_cal():
    s = datetime.combine(date.today() + timedelta(days=1), datetime.min.time()).replace(hour=10)
    return cal(f"""UID:t1
SUMMARY:Team sync
DTSTART:{stamp(s)}
DTEND:{stamp(s + timedelta(minutes=45))}
LOCATION:https://meet.google.com/abc-defg-hij""", """UID:a1
SUMMARY:Holiday
DTSTART;VALUE=DATE:""" + date.today().strftime("%Y%m%d"))


@pytest.fixture
def cal_ctx(settings, ctx, registry, monkeypatch):
    from jarvis.tools import calendar_sync as cal_

    ctx.settings = settings
    monkeypatch.setattr(cal_, "fetch", lambda url, get=None: tomorrow_cal())
    return ctx


def test_calendar_events_days_next_week_and_free(cal_ctx, registry):
    out = registry.call("calendar_events", {}, cal_ctx)
    assert out["day"] == "today" and [e["title"] for e in out["events"]] == ["Holiday"]
    assert out["events"][0]["when"] == "all day" and "_safety" in out  # invite titles come from other people
    out = registry.call("calendar_events", {"day": "kal"}, cal_ctx)
    assert [e["title"] for e in out["events"]] == ["Team sync"] and out["events"][0]["when"] == "10:00 AM - 10:45 AM"
    assert out["events"][0]["join_link"].startswith("https://meet.google.com")
    assert out["free"] == ["9:00 AM - 10:00 AM", "10:45 AM - 9:00 PM"]
    nxt = registry.call("calendar_events", {"day": "next"}, cal_ctx)["next"]
    assert nxt["title"] == "Team sync" and nxt["day"] and nxt["starts_in"].endswith(("h", "min"))
    assert registry.call("calendar_events", {"day": "week"}, cal_ctx)["count"] == 2
    assert "Nothing on the calendar" in registry.call("calendar_events", {"day": "parso"}, cal_ctx)["result"]
    assert "error" in registry.call("calendar_events", {"day": "blue moon"}, cal_ctx)
    assert "next 60 days" in registry.call("calendar_events", {"day": "2030-01-01"}, cal_ctx)["error"]


def test_calendar_not_connected(ctx, registry, isolated_home):
    assert "Settings > Features > Calendar" in registry.call("calendar_events", {}, ctx)["error"]
    assert "error" in registry.call("sync_calendar", {}, ctx)


def test_free_slots():
    from jarvis.tools.calendar_sync import free_slots

    day = date(2026, 11, 10)
    at = lambda h, m=0: datetime(2026, 11, 10, h, m).timestamp()  # noqa: E731
    busy = [{"start": at(10), "end": at(11), "all_day": False}, {"start": at(10, 30), "end": at(12), "all_day": False},
            {"start": at(12, 15), "end": at(13), "all_day": False}, {"start": at(0), "end": at(23), "all_day": True}]
    assert free_slots(busy, day) == ["9:00 AM - 10:00 AM", "1:00 PM - 9:00 PM"]


# -- reminders --------------------------------------------------------------------------------
def test_meeting_reminder_once_not_for_all_day(settings):
    from jarvis.state import StateStore
    from jarvis.tools import calendar_sync as cal_

    cal_.sync(settings, get=lambda *a, **k: Resp(soon_cal(8)))
    st = StateStore(cal_.data_file().path.parent / "state.json")
    first = cal_.reminders_due(settings, st)
    assert [e["title"] for e in first] == ["Design review"]
    assert cal_.reminders_due(settings, st) == []  # only once
    settings.calendar_reminder_minutes = 5
    st.set("calendar_reminded", [])
    assert cal_.reminders_due(settings, st) == []  # 8 min away, 5 min lead
    settings.calendar_reminder_minutes = 0
    assert cal_.reminders_due(settings, st, time.time() + 6 * 60) == []


def test_background_announces_meetings(settings, monkeypatch):
    from jarvis.app import create_app
    from jarvis.background import BackgroundServices
    from jarvis.tools import calendar_sync as cal_

    monkeypatch.setattr(cal_, "fetch", lambda url, get=None: soon_cal(9, "Client call"))
    app = create_app(settings)
    said, nudges = [], []
    app.events.subscribe(lambda k, d: nudges.append(d) if k == "nudge" else None)
    bg = BackgroundServices(app, said.append)
    assert [e["title"] for e in bg.calendar_tick()] == ["Client call"]
    assert "Client call" in said[0] and "join link" in said[0] and nudges[0]["kind"] == "meeting"
    assert bg.calendar_tick() == [] and len(said) == 1


def test_today_strip_shows_next_meeting(settings, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from jarvis.app import create_app
    from jarvis.tools import calendar_sync as cal_
    from jarvis.ui.today import TodayStrip

    qt = QApplication.instance() or QApplication(sys.argv)
    cal_.sync(settings, get=lambda *a, **k: Resp(soon_cal(20, "Interview with Acme Corporation")))
    strip = TodayStrip(create_app(settings), lambda: {}, lambda q: None)
    strip.show()
    qt.processEvents()
    chip = strip.chips["meeting"]
    assert chip.isVisible() and chip.text().startswith(("📅 in 19 min", "📅 in 20 min"))
    assert "Interview with Acm…" in chip.text() and "meet.google.com" in chip.toolTip()
