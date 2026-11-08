"""A small, forgiving iCalendar (.ics) reader: events, repeats, time zones.

expand(text, start, end) -> [Occurrence] for every event instance between two local datetimes.

Handles what Google Calendar, Outlook and Apple Calendar export: folded lines, escaped text, all-day events,
UTC / TZID / floating times (including Windows zone names), RRULE (DAILY/WEEKLY/MONTHLY/YEARLY with
INTERVAL, COUNT, UNTIL, BYDAY incl. "2TU"/"-1FR", BYMONTHDAY, BYMONTH), EXDATE, RDATE, moved or cancelled
single instances (RECURRENCE-ID) and cancelled events. Broken lines are skipped, never fatal.
"""

from __future__ import annotations

import calendar as _cal
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo

MAX_INSTANCES = 3000  # per event, so a broken rule can't run forever
MAX_PERIODS = 40_000
WEEKDAYS = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}
# Outlook / Exchange write Windows zone names.
WINDOWS_ZONES = {
    "India Standard Time": "Asia/Kolkata", "Sri Lanka Standard Time": "Asia/Colombo",
    "Pakistan Standard Time": "Asia/Karachi", "Bangladesh Standard Time": "Asia/Dhaka",
    "Nepal Standard Time": "Asia/Kathmandu", "Arabian Standard Time": "Asia/Dubai", "Arab Standard Time": "Asia/Riyadh",
    "Singapore Standard Time": "Asia/Singapore", "China Standard Time": "Asia/Shanghai",
    "Tokyo Standard Time": "Asia/Tokyo", "Korea Standard Time": "Asia/Seoul", "SE Asia Standard Time": "Asia/Bangkok",
    "AUS Eastern Standard Time": "Australia/Sydney", "W. Australia Standard Time": "Australia/Perth",
    "New Zealand Standard Time": "Pacific/Auckland", "GMT Standard Time": "Europe/London",
    "Greenwich Standard Time": "Atlantic/Reykjavik", "W. Europe Standard Time": "Europe/Berlin",
    "Romance Standard Time": "Europe/Paris", "Central Europe Standard Time": "Europe/Budapest",
    "Central European Standard Time": "Europe/Warsaw", "E. Europe Standard Time": "Europe/Chisinau",
    "FLE Standard Time": "Europe/Kiev", "GTB Standard Time": "Europe/Bucharest", "Russian Standard Time": "Europe/Moscow",
    "Turkey Standard Time": "Europe/Istanbul", "Israel Standard Time": "Asia/Jerusalem",
    "Egypt Standard Time": "Africa/Cairo", "South Africa Standard Time": "Africa/Johannesburg",
    "E. Africa Standard Time": "Africa/Nairobi", "Eastern Standard Time": "America/New_York",
    "Central Standard Time": "America/Chicago", "Mountain Standard Time": "America/Denver",
    "US Mountain Standard Time": "America/Phoenix", "Pacific Standard Time": "America/Los_Angeles",
    "Alaskan Standard Time": "America/Anchorage", "Hawaiian Standard Time": "Pacific/Honolulu",
    "Atlantic Standard Time": "America/Halifax", "Canada Central Standard Time": "America/Regina",
    "E. South America Standard Time": "America/Sao_Paulo", "UTC": "UTC", "Coordinated Universal Time": "UTC",
}


@dataclass
class Occurrence:
    uid: str
    title: str
    start: datetime  # local, timezone-aware
    end: datetime
    all_day: bool
    location: str = ""
    link: str = ""
    calendar: str = ""


# -- lines ------------------------------------------------------------------------------------
def unfold(text: str) -> list[str]:
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for line in text.split("\n"):
        if line[:1] in (" ", "\t") and out:
            out[-1] += line[1:]
        elif line.strip():
            out.append(line)
    return out


def parse_line(line: str) -> tuple[str, dict[str, str], str] | None:
    """'DTSTART;TZID="India Standard Time":20261109T153000' -> ('DTSTART', {'TZID': ...}, '20261109T153000')."""
    in_quote, colon = False, -1
    for i, ch in enumerate(line):
        if ch == '"':
            in_quote = not in_quote
        elif ch == ":" and not in_quote:
            colon = i
            break
    if colon <= 0:
        return None
    head, value = line[:colon], line[colon + 1:]
    parts = re.findall(r'(?:[^;"]|"[^"]*")+', head)
    if not parts:
        return None
    params = {}
    for p in parts[1:]:
        k, _, v = p.partition("=")
        params[k.strip().upper()] = v.strip().strip('"')
    return parts[0].strip().upper(), params, value


def unescape(value: str) -> str:
    return re.sub(r"\\([\\;,nN])", lambda m: "\n" if m.group(1) in "nN" else m.group(1), value).strip()


def components(text: str) -> tuple[list[dict], dict[str, dict], str]:
    """(events, timezones by TZID, calendar name). Each component maps NAME -> [(params, value)]."""
    events, zones, name = [], {}, ""
    stack: list[tuple[str, dict]] = []
    for line in unfold(text):
        parsed = parse_line(line)
        if not parsed:
            continue
        key, params, value = parsed
        if key == "BEGIN":
            stack.append((value.strip().upper(), {}))
            continue
        if key == "END":
            if not stack:
                continue
            kind, comp = stack.pop()
            if kind == "VEVENT":
                events.append(comp)
            elif kind == "VTIMEZONE" and comp.get("TZID"):
                zones[comp["TZID"][0][1].strip()] = comp
            elif kind in ("STANDARD", "DAYLIGHT") and stack and stack[-1][0] == "VTIMEZONE":
                stack[-1][1].setdefault(kind, []).append(({}, comp))
            continue
        if stack:
            stack[-1][1].setdefault(key, []).append((params, value))
        elif key == "X-WR-CALNAME":
            name = unescape(value)
    if stack and stack[-1][0] == "VEVENT":  # a cut-off file: keep the last event if it looks complete
        events.append(stack[-1][1])
    return events, zones, name


def calendar_name(text: str) -> str:
    for line in unfold(text)[:60]:
        p = parse_line(line)
        if p and p[0] == "X-WR-CALNAME":
            return unescape(p[2])
    return ""


# -- time zones ---------------------------------------------------------------------------------
def _offset(text: str) -> timedelta | None:
    m = re.fullmatch(r"([+-])(\d{2})(\d{2})(\d{2})?", text.strip())
    if not m:
        return None
    sign = -1 if m.group(1) == "-" else 1
    return sign * timedelta(hours=int(m.group(2)), minutes=int(m.group(3)), seconds=int(m.group(4) or 0))


def zone(tzid: str, zones: dict[str, dict] | None = None) -> tzinfo | None:
    """A tzinfo for a TZID; None = floating (the computer's own time)."""
    if not tzid:
        return None
    from zoneinfo import ZoneInfo

    tzid = tzid.strip().strip('"')
    candidates = [tzid, WINDOWS_ZONES.get(tzid, "")]
    m = re.search(r"([A-Za-z_]+/[A-Za-z_\-]+(?:/[A-Za-z_\-]+)?)$", tzid)  # '/mozilla.org/.../Europe/Berlin'
    if m:
        candidates.append(m.group(1))
    for name in candidates:
        if not name:
            continue
        try:
            return ZoneInfo(name)
        except Exception:
            continue
    if tzid.upper() in ("Z", "UTC", "GMT", "ETC/UTC"):
        return timezone.utc
    comp = (zones or {}).get(tzid)
    if comp:  # unknown name: use the zone's standard offset from the file (no daylight saving)
        for _, sub in comp.get("STANDARD", []) + comp.get("DAYLIGHT", []):
            off = _offset((sub.get("TZOFFSETTO") or [({}, "")])[0][1])
            if off is not None:
                return timezone(off, tzid)
    return None


def parse_value(value: str, params: dict[str, str], zones: dict) -> tuple[datetime | date, tzinfo | None] | None:
    """(naive wall time or date, its zone). None when unreadable."""
    v = value.strip()
    try:
        if params.get("VALUE", "").upper() == "DATE" or re.fullmatch(r"\d{8}", v):
            return datetime.strptime(v[:8], "%Y%m%d").date(), None
        utc = v.endswith("Z")
        v = v.rstrip("Z")
        dt = datetime.strptime(v[:15], "%Y%m%dT%H%M%S") if len(v) >= 15 else datetime.strptime(v[:13], "%Y%m%dT%H%M")
    except ValueError:
        return None
    if utc:
        return dt, timezone.utc
    return dt, zone(params.get("TZID", ""), zones)


def _local(wall: datetime, tz: tzinfo | None) -> datetime:
    """Wall time in `tz` -> aware local time (floating times are already local)."""
    if tz is None:
        return wall.astimezone()
    return wall.replace(tzinfo=tz).astimezone()


def parse_duration(text: str) -> timedelta | None:
    m = re.fullmatch(r"([+-])?P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", text.strip())
    if not m or text.strip() in ("P", "PT"):
        return None
    w, d, h, mi, s = (int(x or 0) for x in m.groups()[1:])
    out = timedelta(weeks=w, days=d, hours=h, minutes=mi, seconds=s)
    return -out if m.group(1) == "-" else out


# -- repeats ----------------------------------------------------------------------------------
def parse_rule(text: str) -> dict[str, str]:
    rule = {}
    for part in text.split(";"):
        k, _, v = part.partition("=")
        if k.strip():
            rule[k.strip().upper()] = v.strip().upper()
    return rule


def _byday(rule: dict) -> list[tuple[int, int]]:
    """'2TU,-1FR,MO' -> [(2, 1), (-1, 4), (0, 0)]  (0 = every such weekday)."""
    out = []
    for item in filter(None, rule.get("BYDAY", "").split(",")):
        m = re.fullmatch(r"([+-]?\d{1,2})?(MO|TU|WE|TH|FR|SA|SU)", item.strip())
        if m:
            out.append((int(m.group(1) or 0), WEEKDAYS[m.group(2)]))
    return out


def _ints(rule: dict, key: str) -> list[int]:
    out = []
    for x in filter(None, rule.get(key, "").split(",")):
        try:
            out.append(int(x))
        except ValueError:
            pass
    return out


def _month_days(year: int, month: int, rule: dict, default_day: int) -> list[int]:
    last = _cal.monthrange(year, month)[1]
    days = set()
    for d in _ints(rule, "BYMONTHDAY"):
        d = d if d > 0 else last + 1 + d
        if 1 <= d <= last:
            days.add(d)
    for n, wd in _byday(rule):
        all_wd = [d for d in range(1, last + 1) if date(year, month, d).weekday() == wd]
        if n == 0:
            days.update(all_wd)
        elif -len(all_wd) <= n <= len(all_wd) and n != 0:
            days.add(all_wd[n - 1] if n > 0 else all_wd[n])
    if not rule.get("BYMONTHDAY") and not rule.get("BYDAY") and default_day <= last:
        days.add(default_day)  # Jan 31 monthly: months without a 31st are skipped (as the standard says)
    return sorted(days)


def instances(first: datetime, rule: dict, until_wall: datetime | None, window_end: datetime,
              window_start: datetime | None = None) -> list[datetime]:
    """Naive wall-clock starts of a repeating event, in order, up to window_end (inclusive).

    Without COUNT, starts before window_start are skipped (and whole periods jumped), so an event repeating
    since 2005 is as cheap as a new one."""
    freq = rule.get("FREQ", "")
    try:
        interval = max(1, int(rule.get("INTERVAL", "1") or 1))
        count = int(rule["COUNT"]) if rule.get("COUNT") else None
    except ValueError:
        return [first]
    months = set(_ints(rule, "BYMONTH"))
    weekdays = [wd for _, wd in _byday(rule)]
    t = first.time()
    out: list[datetime] = []
    limit = until_wall if until_wall is not None else window_end
    stop = min(limit, window_end)

    jump = count is None and window_start is not None and window_start > first

    def take(cands) -> bool:
        """Add candidates in order; False once we're done."""
        for c in cands:
            if c < first or jump and c < window_start:
                continue
            if until_wall is not None and c > until_wall:
                return False
            if count is not None and len(out) >= count:
                return False
            if c > window_end:
                return False
            out.append(c)
            if len(out) >= MAX_INSTANCES:
                return False
        return True

    def n0(span_days: float) -> int:
        return max(0, int((window_start - first).days / span_days) - 1) if jump else 0

    if freq == "DAILY":
        for n in range(n0(interval), n0(interval) + MAX_PERIODS):
            c = first + timedelta(days=n * interval)
            if c > stop:
                break
            if months and c.month not in months or weekdays and c.weekday() not in weekdays:
                continue
            if not take([c]):
                break
    elif freq == "WEEKLY":
        wkst = WEEKDAYS.get(rule.get("WKST", "MO"), 0)
        week0 = first.date() - timedelta(days=(first.weekday() - wkst) % 7)
        days = sorted({(wd - wkst) % 7 for wd in (weekdays or [first.weekday()])})
        for n in range(n0(7 * interval), n0(7 * interval) + MAX_PERIODS):
            start = week0 + timedelta(weeks=n * interval)
            if datetime.combine(start, time()) > stop:
                break
            cands = [datetime.combine(start + timedelta(days=d), t) for d in days]
            if months:
                cands = [c for c in cands if c.month in months]
            if not take(cands):
                break
    elif freq == "MONTHLY":
        for n in range(n0(31 * interval), n0(31 * interval) + MAX_PERIODS):
            m = first.month - 1 + n * interval
            y, mo = first.year + m // 12, m % 12 + 1
            if datetime(y, mo, 1) > stop:
                break
            if months and mo not in months:
                continue
            if not take([datetime.combine(date(y, mo, d), t) for d in _month_days(y, mo, rule, first.day)]):
                break
    elif freq == "YEARLY":
        for n in range(n0(366 * interval), n0(366 * interval) + MAX_PERIODS):
            y = first.year + n * interval
            if datetime(y, 1, 1) > stop:
                break
            cands = []
            for mo in sorted(months or {first.month}):
                if rule.get("BYMONTHDAY") or rule.get("BYDAY"):
                    cands += [datetime.combine(date(y, mo, d), t) for d in _month_days(y, mo, rule, first.day)]
                elif first.day <= _cal.monthrange(y, mo)[1]:  # Feb 29 only in leap years
                    cands.append(datetime.combine(date(y, mo, first.day), t))
            if not take(sorted(cands)):
                break
    else:
        return [first]
    return out


# -- events -----------------------------------------------------------------------------------
LINK = re.compile(r"https://[^\s<>\"']*(?:meet\.google\.com|zoom\.us|teams\.microsoft\.com|teams\.live\.com|"
                  r"webex\.com|whereby\.com|meet\.jit\.si)[^\s<>\"']*", re.I)


def _first(comp: dict, key: str) -> tuple[dict, str] | None:
    vals = comp.get(key)
    return vals[0] if vals else None


def _text(comp: dict, key: str) -> str:
    v = _first(comp, key)
    return unescape(v[1]) if v else ""


def _as_wall(value: datetime | date, tz: tzinfo | None, target_tz: tzinfo | None) -> datetime:
    """A parsed value as naive wall time in the event's own zone (for UNTIL / EXDATE comparisons)."""
    if isinstance(value, datetime):
        if tz is not None and target_tz is not None and tz is not target_tz:
            return value.replace(tzinfo=tz).astimezone(target_tz).replace(tzinfo=None)
        if tz is not None and target_tz is None:
            return value.replace(tzinfo=tz).astimezone().replace(tzinfo=None)
        return value
    return datetime.combine(value, time(23, 59, 59))


def expand(text: str, start: datetime, end: datetime, calendar: str = "") -> list[Occurrence]:
    """Every event instance overlapping [start, end) (local naive or aware datetimes), sorted."""
    start = start if start.tzinfo else start.astimezone()
    end = end if end.tzinfo else end.astimezone()
    events, zones, _ = components(text)
    overrides: dict[str, dict[str, dict]] = {}  # uid -> {wall-time key: event}
    plain = []
    for ev in events:
        rid = _first(ev, "RECURRENCE-ID")
        if rid:
            uid = _text(ev, "UID")
            parsed = parse_value(rid[1], rid[0], zones)
            if uid and parsed:
                overrides.setdefault(uid, {})[_stamp(*parsed)] = ev
            continue
        plain.append(ev)
    out: list[Occurrence] = []
    for ev in plain:
        try:
            out.extend(_expand_event(ev, zones, start, end, overrides.get(_text(ev, "UID"), {}), calendar))
        except Exception:  # one odd event never breaks the calendar
            continue
    # moved instances whose master is missing (or that moved into the window)
    seen = {(o.uid, o.start) for o in out}
    for uid, moved in overrides.items():
        for ev in moved.values():
            for occ in _single(ev, zones, calendar):
                if (occ.uid, occ.start) not in seen and occ.end > start and occ.start < end:
                    out.append(occ)
                    seen.add((occ.uid, occ.start))
    return sorted(out, key=lambda o: (o.start, o.title))


def _stamp(value: datetime | date, tz: tzinfo | None) -> str:
    """The same moment gives the same key, whatever zone it was written in."""
    if isinstance(value, datetime):
        return str(int(_local(value, tz).timestamp()))
    return "D" + value.strftime("%Y%m%d")


def _cancelled(ev: dict) -> bool:
    return _text(ev, "STATUS").upper() == "CANCELLED"


def _span(ev: dict, zones: dict):
    """(first start value, zone, all_day, duration) or None."""
    ds = _first(ev, "DTSTART")
    if not ds:
        return None
    parsed = parse_value(ds[1], ds[0], zones)
    if not parsed:
        return None
    first, tz = parsed
    all_day = not isinstance(first, datetime)
    duration = None
    de = _first(ev, "DTEND")
    if de:
        pe = parse_value(de[1], de[0], zones)
        if pe:
            if all_day and not isinstance(pe[0], datetime):
                duration = timedelta(days=(pe[0] - first).days)
            elif not all_day and isinstance(pe[0], datetime):
                duration = _local(pe[0], pe[1]) - _local(first, tz)
    if duration is None and _first(ev, "DURATION"):
        duration = parse_duration(_first(ev, "DURATION")[1])
    if duration is None or duration < timedelta(0):
        duration = timedelta(days=1) if all_day else timedelta(0)
    if all_day and duration < timedelta(days=1):
        duration = timedelta(days=1)
    return first, tz, all_day, duration


def _occ(ev: dict, wall, tz, all_day: bool, duration: timedelta, calendar: str) -> Occurrence:
    if all_day:
        d = wall if isinstance(wall, date) and not isinstance(wall, datetime) else wall.date()
        s = datetime.combine(d, time()).astimezone()
        e = datetime.combine(d + duration, time()).astimezone()
    else:
        s = _local(wall, tz)
        e = s + duration
    blob = " ".join(_text(ev, k) for k in ("LOCATION", "URL", "DESCRIPTION")) + " " + _text(ev, "X-GOOGLE-CONFERENCE")
    link = LINK.search(blob)
    return Occurrence(uid=_text(ev, "UID") or _text(ev, "SUMMARY"), title=_text(ev, "SUMMARY") or "(No title)",
                      start=s, end=e, all_day=all_day, location=_text(ev, "LOCATION")[:120],
                      link=link.group(0).rstrip(".,)>") if link else "", calendar=calendar)


def _single(ev: dict, zones: dict, calendar: str) -> list[Occurrence]:
    if _cancelled(ev):
        return []
    span = _span(ev, zones)
    if not span:
        return []
    first, tz, all_day, duration = span
    return [_occ(ev, first, tz, all_day, duration, calendar)]


def _expand_event(ev: dict, zones: dict, start: datetime, end: datetime, moved: dict[str, dict],
                  calendar: str) -> list[Occurrence]:
    if _cancelled(ev):
        return []
    span = _span(ev, zones)
    if not span:
        return []
    first, tz, all_day, duration = span
    rr = _first(ev, "RRULE")
    if not rr:
        occ = _occ(ev, first, tz, all_day, duration, calendar)
        return [occ] if occ.end > start and occ.start < end else []
    rule = parse_rule(rr[1])
    first_wall = first if isinstance(first, datetime) else datetime.combine(first, time())
    until_wall = None
    if rule.get("UNTIL"):
        pu = parse_value(rule["UNTIL"], {}, zones)
        if pu:
            until_wall = _as_wall(pu[0], pu[1], tz) if not all_day else (
                datetime.combine(pu[0] if not isinstance(pu[0], datetime) else pu[0].date(), time(23, 59, 59)))
    # the window in the event's own wall time (a day of slack on each side for zone differences)
    win_end = (end.astimezone(tz) if tz else end.astimezone()).replace(tzinfo=None) + timedelta(days=1)
    win_start = (start.astimezone(tz) if tz else start.astimezone()).replace(tzinfo=None) - timedelta(days=1)
    win_start -= duration  # an instance that started before the window may still be going on
    starts = instances(first_wall, rule, until_wall, win_end, win_start)
    for _, value in ev.get("RDATE", []):
        for v in value.split(","):
            p = parse_value(v, {"TZID": _first(ev, "DTSTART")[0].get("TZID", "")}, zones)
            if p:
                w = p[0] if isinstance(p[0], datetime) else datetime.combine(p[0], first_wall.time())
                starts.append(_as_wall(w, p[1], tz) if isinstance(p[0], datetime) else w)
    skipped = set()
    for params, value in ev.get("EXDATE", []):
        for v in value.split(","):
            p = parse_value(v, params, zones)
            if p:
                skipped.add(_stamp(p[0].date() if all_day and isinstance(p[0], datetime) else p[0], p[1]))
    out = []
    for wall in sorted(set(starts)):
        k = _stamp(wall.date(), None) if all_day else _stamp(wall, tz)
        if k in skipped or k in moved:
            continue  # skipped, or replaced by its moved/cancelled copy (added by expand)
        occ = _occ(ev, wall.date() if all_day else wall, tz, all_day, duration, calendar)
        if occ.end > start and occ.start < end:
            out.append(occ)
    return out
