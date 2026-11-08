"""Calendar sync from private iCal links (Google Calendar, Outlook, Apple, anything that exports .ics).

- Read-only: Jarvis reads your calendar, it can't change it.
- The links work like passwords: they're encrypted in config.json (Windows DPAPI), never logged, never shown
  to the AI and never put in error messages.
- A copy of the next 60 days is kept in ~/.jarvis/calendar.json so it works offline and starts fast.
  If one calendar fails, its last good copy is kept and marked as old.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import date, datetime, timedelta
from urllib.parse import urlsplit

from jarvis.datafile import store
from jarvis.tools import tool

SYNC_EVERY = 15 * 60
DAYS_AHEAD = 60
MAX_BYTES = 8 * 1024 * 1024
DEFAULT = {"events": [], "sources": {}, "synced": 0, "links": ""}
SETUP_HELP = ("No calendar is connected yet. Open Settings > Features > Calendar and paste your calendar's "
              "private iCal link. Google Calendar: Settings > (your calendar) > Integrate calendar > "
              "'Secret address in iCal format'. Outlook: Settings > Calendar > Shared calendars > Publish a "
              "calendar > ICS link.")
# urllib3 logs request paths at debug level; private links must never reach a log file
logging.getLogger("urllib3").setLevel(logging.WARNING)
_mem: dict = {"mtime": None, "data": None}


def data_file():
    return store("calendar", DEFAULT)


def links(settings) -> list[str]:
    """The configured links, cleaned: webcal:// -> https://, blanks and duplicates removed."""
    out = []
    for raw in getattr(settings, "calendar_urls", []) or []:
        for u in re.split(r"[\s,]+", raw or ""):
            u = u.strip().strip("<>\"'")
            if not u:
                continue
            if u.lower().startswith("webcal://"):
                u = "https://" + u[9:]
            if u not in out:
                out.append(u)
    return out


def source_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]


def links_key(urls: list[str]) -> str:
    return hashlib.sha256("|".join(urls).encode("utf-8")).hexdigest()[:16]


def check_link(url: str) -> str | None:
    """A reason the link can't be used, or None."""
    parts = urlsplit(url)
    if parts.scheme not in ("https", "http") or not parts.hostname:
        return "That isn't a web link. Paste the iCal / ICS address (it starts with https:// or webcal://)."
    if "calendar.google.com" in parts.hostname and "/ical/" not in parts.path:
        return ("That's the normal Google Calendar page link. Use 'Secret address in iCal format' from the "
                "calendar's settings instead.")
    return None


def fetch(url: str, get=None) -> str:
    """The .ics text. Raises ValueError with a friendly message that never contains the link."""
    problem = check_link(url)
    if problem:
        raise ValueError(problem)
    if get is None:
        import requests

        get = requests.get
    try:
        resp = get(url, timeout=20, stream=True, headers={"User-Agent": "Jarvis-Assistant/1.0 (calendar)"})
    except Exception as exc:
        name = type(exc).__name__
        if "Timeout" in name:
            raise ValueError("The calendar server didn't answer in time.") from None
        if "SSL" in name:
            raise ValueError("The calendar server's security certificate wasn't accepted.") from None
        raise ValueError("Couldn't reach the calendar (no internet?).") from None
    status = getattr(resp, "status_code", 200)
    if status in (401, 403, 404, 410):
        raise ValueError("The calendar link doesn't work any more (it may have been reset). "
                         "Copy the private iCal link again into Settings.")
    if status >= 400:
        raise ValueError(f"The calendar server had a problem (error {status}). I'll try again later.")
    chunks, size = [], 0
    try:
        for chunk in resp.iter_content(64 * 1024):
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError("That calendar is too big to read (over 8 MB).")
            chunks.append(chunk)
    finally:
        close = getattr(resp, "close", None)
        if close:
            close()
    raw = b"".join(chunks)
    text = raw.decode(getattr(resp, "encoding", None) or "utf-8", errors="replace") if raw else ""
    if "BEGIN:VCALENDAR" not in text[:4096].upper():
        raise ValueError("That link didn't return a calendar. Make sure it's the iCal / ICS address, "
                         "not the web page of the calendar.")
    return text


def _event_row(o, sid: str) -> dict:
    return {"uid": o.uid[:200], "title": o.title[:200], "start": o.start.timestamp(), "end": o.end.timestamp(),
            "all_day": o.all_day, "location": o.location, "link": o.link, "calendar": o.calendar[:60],
            "source": sid}


def sync(settings, now: datetime | None = None, get=None) -> dict:
    """Fetch every linked calendar and store the next DAYS_AHEAD days. Returns a status summary."""
    from jarvis.ics import calendar_name, expand

    now = now or datetime.now()
    urls = links(settings)
    f = data_file()
    if not urls:
        if f.exists():
            f.delete()
        return {"calendars": 0, "events": 0, "errors": []}
    old = f.load()
    start = datetime.combine(now.date() - timedelta(days=1), datetime.min.time())
    end = datetime.combine(now.date() + timedelta(days=DAYS_AHEAD), datetime.min.time())
    rows, sources, errors = [], {}, []
    for i, url in enumerate(urls, 1):
        sid = source_id(url)
        prev = old.get("sources", {}).get(sid, {})
        try:
            text = fetch(url, get)
            name = calendar_name(text) or prev.get("name") or f"Calendar {i}"
            occ = expand(text, start, end, calendar=name)
            rows += [_event_row(o, sid) for o in occ]
            sources[sid] = {"name": name, "ok": True, "error": "", "count": len(occ), "synced": time.time()}
        except Exception as exc:
            msg = str(exc) if isinstance(exc, ValueError) else f"Couldn't read the calendar ({type(exc).__name__})."
            errors.append(f"{prev.get('name') or f'Calendar {i}'}: {msg}")
            rows += [e for e in old.get("events", []) if e.get("source") == sid]  # keep the last good copy
            sources[sid] = {**prev, "name": prev.get("name") or f"Calendar {i}", "ok": False, "error": msg}
    seen, unique = set(), []
    for r in sorted(rows, key=lambda r: (r["start"], r["title"])):
        key = (r["uid"], r["start"])
        if key not in seen:  # the same meeting in two linked calendars
            seen.add(key)
            unique.append(r)
    f.save({"events": unique, "sources": sources, "synced": time.time(), "links": links_key(urls)})
    return {"calendars": len(urls), "events": len(unique), "errors": errors}


def due(settings, now: float | None = None) -> bool:
    urls = links(settings)
    if not urls:
        return data_file().exists()  # links removed: clear the copy
    data = data_file().load()
    now = now or time.time()
    return data.get("links") != links_key(urls) or now - data.get("synced", 0) >= SYNC_EVERY \
        or data.get("synced", 0) > now + 60  # the clock went back


def events(settings=None) -> list[dict]:
    """Stored events (cached in memory until the file changes)."""
    f = data_file()
    try:
        mtime = f.path.stat().st_mtime
    except OSError:
        return []
    if _mem["mtime"] != mtime:
        _mem["mtime"], _mem["data"] = mtime, f.load()
    if settings is not None and _mem["data"].get("links") != links_key(links(settings)):
        return []  # copy from links that were removed or changed
    return _mem["data"].get("events", [])


def between(settings, start: float, end: float) -> list[dict]:
    return [e for e in events(settings) if e["end"] > start and e["start"] < end]


def next_event(settings, now: float | None = None, within: float = 12 * 3600) -> dict | None:
    """The meeting happening now or next (timed events only)."""
    now = now or time.time()
    for e in events(settings):
        if not e["all_day"] and e["end"] > now and e["start"] < now + within:
            return e
    return None


def reminders_due(settings, state, now: float | None = None) -> list[dict]:
    """Timed events starting within the reminder window that haven't been reminded yet (marks them)."""
    minutes = getattr(settings, "calendar_reminder_minutes", 10)
    if not minutes:
        return []
    now = now or time.time()
    done = list(state.get("calendar_reminded", []) or [])
    out = []
    for e in events(settings):
        lead = e["start"] - now
        if e["all_day"] or not (0 < lead <= minutes * 60 + 30):
            continue
        key = f"{e['uid']}@{int(e['start'])}"
        if key in done:
            continue
        done.append(key)
        out.append(e)
    if out:
        state.set("calendar_reminded", done[-300:])
    return out


def _clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%I:%M %p").lstrip("0")


def describe(e: dict) -> dict:
    if e["all_day"]:
        days = max(1, round((e["end"] - e["start"]) / 86400))
        when = "all day" + (f" ({days} days)" if days > 1 else "")
    else:
        when = f"{_clock(e['start'])} - {_clock(e['end'])}" if e["end"] > e["start"] else _clock(e["start"])
    row = {"when": when, "title": e["title"]}
    if datetime.fromtimestamp(e["start"]).date() != date.today():
        row["day"] = datetime.fromtimestamp(e["start"]).strftime("%a %d %b")
    for k in ("location", "link", "calendar"):
        if e.get(k):
            row[k if k != "link" else "join_link"] = e[k]
    return row


def free_slots(busy: list[dict], day: date, now: float | None = None, start_h: int = 9, end_h: int = 21) -> list[str]:
    """Free gaps of 30+ minutes in the working part of a day."""
    lo = datetime.combine(day, datetime.min.time()).replace(hour=start_h).timestamp()
    hi = datetime.combine(day, datetime.min.time()).replace(hour=end_h).timestamp()
    if now:
        lo = max(lo, now)
    spans = sorted((max(e["start"], lo), min(e["end"], hi)) for e in busy if not e["all_day"])
    out, cur = [], lo
    for s, e in spans:
        if s - cur >= 1800:
            out.append(f"{_clock(cur)} - {_clock(s)}")
        cur = max(cur, e)
    if hi - cur >= 1800:
        out.append(f"{_clock(cur)} - {_clock(hi)}")
    return out


def _age(ts: float) -> str:
    mins = int((time.time() - ts) / 60)
    return "just now" if mins < 1 else f"{mins} min ago" if mins < 60 else f"{mins // 60} h ago"


@tool(
    "calendar_events",
    "The user's calendar (read-only): meetings and events for a day or the week, the next meeting, and free "
    "time. day: today, tomorrow/kal, parso, a weekday, a date, 'week' or 'next'.",
    {"day": {"type": "string", "description": "today (default), tomorrow, kal, monday, 12 nov, week, next"}},
    slow=True,
)
def calendar_events(ctx, day: str = "today"):
    s = ctx.settings
    if not links(s):
        return {"error": SETUP_HELP}
    if due(s) or not events(s):
        try:
            sync(s)
        except Exception as exc:
            ctx.events and ctx.events.emit("log", text=f"[calendar] {type(exc).__name__}")
    data = data_file().load()
    problems = [f"{v.get('name')}: {v.get('error')}" for v in data.get("sources", {}).values() if not v.get("ok")]
    extra = {"synced": _age(data.get("synced", 0))} if data.get("synced") else {}
    if problems:
        extra["warning"] = "Some calendars couldn't be updated, showing the last copy: " + " | ".join(problems)
    now = time.time()
    d = (day or "today").strip().lower()
    if d in ("next", "next meeting", "agla", "agli meeting"):
        e = next_event(s, now, within=DAYS_AHEAD * 86400)
        if not e:
            return {"result": f"No upcoming meetings in the next {DAYS_AHEAD} days.", **extra}
        mins = int((e["start"] - now) / 60)
        row = describe(e)
        row["starts_in"] = "now (in progress)" if mins < 0 else f"{mins} min" if mins < 120 else f"{mins // 60} h"
        return {"next": row, **extra}
    if d in ("week", "this week", "is hafte", "next 7 days"):
        start = datetime.combine(date.today(), datetime.min.time())
        rows = between(s, start.timestamp(), (start + timedelta(days=7)).timestamp())
        days: dict[str, list] = {}
        for e in rows:
            days.setdefault(datetime.fromtimestamp(max(e["start"], start.timestamp())).strftime("%a %d %b"),
                            []).append(describe(e) | {"day": None})
        return {"week": {k: [{x: y for x, y in r.items() if y is not None} for r in v] for k, v in days.items()},
                "count": len(rows), **extra} if rows else {"result": "Nothing on the calendar this week.", **extra}
    from jarvis.dates import parse_day

    target = parse_day(d, past=False)
    if target is None:
        return {"error": f"Couldn't understand the day '{day}'."}
    if target > date.today() + timedelta(days=DAYS_AHEAD):
        return {"error": f"I only keep the next {DAYS_AHEAD} days of the calendar."}
    start = datetime.combine(target, datetime.min.time())
    rows = between(s, start.timestamp(), (start + timedelta(days=1)).timestamp())
    label = "today" if target == date.today() else start.strftime("%A %d %b")
    out = {"day": label, "events": [describe(e) | {} for e in rows], **extra}
    for r in out["events"]:
        r.pop("day", None)
    if target >= date.today():
        out["free"] = free_slots(rows, target, now if target == date.today() else None)
    if not rows:
        out["result"] = f"Nothing on the calendar for {label}."
    return out


@tool("sync_calendar", "Refresh the calendar now (it also refreshes by itself every 15 minutes).", slow=True)
def sync_calendar(ctx):
    if not links(ctx.settings):
        return {"error": SETUP_HELP}
    out = sync(ctx.settings)
    if ctx.events is not None:
        ctx.events.emit("calendar", **out)
    if out["errors"]:
        return {"result": f"Updated {out['calendars'] - len(out['errors'])} of {out['calendars']} calendar(s).",
                "problems": out["errors"]}
    return {"result": f"Calendar updated: {out['events']} events in the next {DAYS_AHEAD} days."}
