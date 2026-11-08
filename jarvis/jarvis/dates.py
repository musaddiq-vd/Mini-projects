"""Understanding everyday dates, in English and in common Hindi words written in Latin letters.

The Hindi word for "yesterday" and "tomorrow" is the same, so the caller says which way to read it:
past=True for things that happened (expenses), past=False for plans (goals, deadlines).
Also understood: today, the day after, "3 days ago", "in 3 weeks", "next month", weekdays, "12 nov",
"12/11", "2026-12-31", "end of month" (and their Hindi equivalents).
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_name) if m})
DAYS = {d.lower(): i for i, d in enumerate(calendar.day_name)}
DAYS.update({d.lower(): i for i, d in enumerate(calendar.day_abbr)})
HINDI_DAYS = {"somvar": 0, "mangalvar": 1, "budhvar": 2, "guruvar": 3, "shukravar": 4, "shanivar": 5, "ravivar": 6,
              "itvar": 6}
UNITS = {"day": 1, "days": 1, "din": 1, "week": 7, "weeks": 7, "hafte": 7, "hafta": 7, "month": 30, "months": 30,
         "mahine": 30, "mahina": 30, "year": 365, "years": 365, "saal": 365}


def add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def parse_day(text: str | None, today: date | None = None, past: bool = True) -> date | None:
    """A date from words; None when it can't be understood. Empty text = today."""
    today = today or date.today()
    t = " ".join((text or "").lower().replace(",", " ").split())
    if not t or t in {"today", "aaj", "now", "abhi"}:
        return today
    if t in {"yesterday"}:
        return today - timedelta(days=1)
    if t in {"tomorrow"}:
        return today + timedelta(days=1)
    if t in {"kal"}:
        return today - timedelta(days=1) if past else today + timedelta(days=1)
    if t in {"parso", "parson"}:
        return today - timedelta(days=2) if past else today + timedelta(days=2)
    if t in {"end of month", "month end", "mahine ke end", "mahine ke aakhir"}:
        return today.replace(day=calendar.monthrange(today.year, today.month)[1])
    if t in {"end of year", "year end", "saal ke end"}:
        return date(today.year, 12, 31)
    if t in {"next week", "agle hafte"}:
        return today + timedelta(days=7)
    if t in {"next month", "agle mahine"}:
        return add_months(today, 1)
    if t in {"next year", "agle saal"}:
        return date(today.year + 1, today.month, min(today.day, 28))
    m = re.fullmatch(r"(\d+)\s+(\w+)\s+(?:ago|pehle|pahle)", t)
    if m and m.group(2) in UNITS:
        n, unit = int(m.group(1)), m.group(2)
        return add_months(today, -n) if UNITS[unit] == 30 else today - timedelta(days=n * UNITS[unit])
    m = re.fullmatch(r"(?:in\s+)?(\d+)\s+(\w+)(?:\s+(?:later|baad|mein|me))?", t)
    if m and m.group(2) in UNITS:
        n, unit = int(m.group(1)), m.group(2)
        return add_months(today, n) if UNITS[unit] == 30 else today + timedelta(days=n * UNITS[unit])
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        return _safe(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})(?:[/.-](\d{2,4}))?", t)  # dd/mm[/yy]
    if m:
        y = int(m.group(3)) if m.group(3) else None
        y = (2000 + y if y and y < 100 else y)
        return _with_year(int(m.group(1)), int(m.group(2)), y, today, past)
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)(?:\s+(\d{4}))?", t) or \
        re.fullmatch(r"([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(\d{4}))?", t)
    if m:
        a, b = m.group(1), m.group(2)
        day_s, mon_s = (a, b) if a.isdigit() else (b, a)
        if mon_s in MONTHS:
            return _with_year(int(day_s), MONTHS[mon_s], int(m.group(3)) if m.group(3) else None, today, past)
    word = t.removeprefix("last ").removeprefix("next ").removeprefix("this ")
    wd = DAYS.get(word, HINDI_DAYS.get(word))
    if wd is not None:
        if t.startswith("last ") or (past and not t.startswith("next ")):
            delta = (today.weekday() - wd) % 7 or (7 if t.startswith("last ") else 0)
            return today - timedelta(days=delta)
        delta = (wd - today.weekday()) % 7 or 7
        return today + timedelta(days=delta)
    return None


def _safe(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _with_year(d: int, m: int, y: int | None, today: date, past: bool) -> date | None:
    if y:
        return _safe(y, m, d)
    guess = _safe(today.year, m, d)
    if guess is None:
        return None
    if past and guess > today:
        return _safe(today.year - 1, m, d)
    if not past and guess < today:
        return _safe(today.year + 1, m, d)
    return guess


def period_range(period: str | None, today: date | None = None) -> tuple[date, date, str]:
    """('today'|'week'|'month'|'last_month'|'year'|'all') -> (start, end inclusive, label)."""
    today = today or date.today()
    p = (period or "month").lower()
    if p in {"today", "day", "aaj"}:
        return today, today, "today"
    if p in {"yesterday", "kal"}:
        y = today - timedelta(days=1)
        return y, y, "yesterday"
    if p in {"week", "this week", "7 days", "hafta"}:
        return today - timedelta(days=6), today, "the last 7 days"
    if p in {"last_month", "last month", "pichhla mahina"}:
        first = add_months(today.replace(day=1), -1)
        return first, today.replace(day=1) - timedelta(days=1), first.strftime("%B")
    if p in {"year", "this year", "saal"}:
        return date(today.year, 1, 1), today, str(today.year)
    if p in {"all", "ever"}:
        return date(2000, 1, 1), today, "all time"
    return today.replace(day=1), today, today.strftime("%B")
