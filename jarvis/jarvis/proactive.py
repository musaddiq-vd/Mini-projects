"""Proactive suggestions: Jarvis notices simple patterns and offers help before being asked.

How it fits together:
- ``Snapshot`` is everything a rule may look at: the time, to-dos, reminders, meetings, battery and disk readings
  (with a short history), habits and when they are usually done, budgets, birthdays, repeated action sequences.
  ``gather`` builds one from the running app; tests build them by hand.
- A rule is a pure function ``Snapshot -> list[Suggestion]`` (see ``RULES``). Rules never change anything.
- ``pick`` scores the candidates and applies the limits: the setting is on, not in focus or practice mode,
  not in quiet hours (unless the rule is meant for the night), at most N per hour, a gap between two, a
  dismissed suggestion is not repeated for a day, and a rule marked "not useful" stays muted.
- ``feedback`` records Accept / Dismiss / Not useful; ``observe`` learns from the tools the user runs (when habits
  get checked, which actions keep coming together). Everything is kept in state under "proactive".

Suggestion text that came from outside (a calendar title) is shown on the card but never spoken or sent to the
model as a request: the Accept request is always built from fixed templates and the user's own data.
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

log = logging.getLogger("jarvis.proactive")

KEY = "proactive"
MIN_GAP = 10 * 60  # seconds between two suggestions
KEY_COOLDOWN = 6 * 3600  # the same suggestion is not shown again for this long
DISMISS_COOLDOWN = 24 * 3600  # ... or this long after Dismiss / Accept
THRESHOLD = 0.4  # lowest score that is shown
MAX_SHOWN = 60
READING_EVERY = 5 * 60  # battery / disk samples
MAX_READINGS = 300
BURST_GAP = 180  # actions closer than this belong to one sequence
WORKFLOW_TOOLS = {"open_app", "open_website", "play_youtube", "volume", "media_control", "dark_mode", "focus_mode",
                  "game_launcher", "brightness", "power_mode"}
OUTCOMES = ("accepted", "dismissed", "not_useful")


@dataclass
class Suggestion:
    rule: str
    key: str  # what makes it "the same suggestion" (rule + subject)
    title: str  # shown on the card
    why: str  # shown by why_did_you_suggest
    action: str = ""  # sent to Jarvis as the user's request on Accept ("" = nothing to do)
    score: float = 0.5  # 0..1 importance before feedback
    speak: str = ""  # a short line Jarvis may say (trusted text only); "" = the title
    icon: str = "💡"
    quiet_ok: bool = False  # may show during quiet hours (but is never spoken then)


@dataclass
class Snapshot:
    now: datetime
    todos: list[dict] = field(default_factory=list)  # open to-dos {id, text, due, priority}
    reminders: list[dict] = field(default_factory=list)  # pending reminders {message, due}
    events: list[dict] = field(default_factory=list)  # calendar events {uid, title, start, end, all_day, link}
    reminded_events: set[str] = field(default_factory=set)
    battery: tuple[float, bool] | None = None  # (percent, plugged in)
    battery_log: list[list[float]] = field(default_factory=list)  # [t, percent, plugged]
    disk_free_gb: float | None = None
    disk_log: list[list[float]] = field(default_factory=list)  # [t, free GB]
    habits: list[dict] = field(default_factory=list)
    habit_times: dict[str, list[int]] = field(default_factory=dict)  # habit -> minutes after midnight
    expenses: list[dict] = field(default_factory=list)
    budgets: dict[str, float] = field(default_factory=dict)
    birthdays: list[tuple[str, str]] = field(default_factory=list)  # (name, "MM-DD")
    idle_seconds: float = 0.0
    bursts: dict[str, dict] = field(default_factory=dict)  # "step|step" -> {count, days}
    routines: dict[str, list[str]] = field(default_factory=dict)
    downloads_loose: int | None = None
    battery_saver_percent: int = 25
    currency: str = "₹"


# -- the rules -------------------------------------------------------------------------------------
def _today(s: Snapshot) -> str:
    return s.now.date().isoformat()


def _has_reminder(s: Snapshot, text: str) -> bool:
    t = " ".join(text.lower().split())
    for r in s.reminders:
        m = " ".join(str(r.get("message", "")).lower().split())
        if m and (t in m or m in t):
            return True
    return False


def _clock(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")


def rule_todo_reminder(s: Snapshot) -> list[Suggestion]:
    """A to-do due today (or an urgent one due tomorrow) with no reminder set."""
    out = []
    tomorrow = (s.now.date() + timedelta(days=1)).isoformat()
    for n in s.todos:
        due, text = n.get("due"), str(n.get("text", "")).strip()
        if not text or n.get("done") or _has_reminder(s, text):
            continue
        if due == _today(s) and 9 <= s.now.hour < 20:
            at = (s.now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            out.append(Suggestion(
                "todo_reminder", f"todo:{n.get('id')}:{due}", f"'{text}' is due today and has no reminder. "
                f"Remind you at {_clock(at)}?", f"The to-do '{text}' is due today ({due}) and no reminder "
                "mentions it.", f"Remind me at {_clock(at)} to: {text}",
                0.8 if n.get("priority") == "high" else 0.65, icon="📝"))
        elif due == tomorrow and n.get("priority") == "high" and s.now.hour >= 18:
            out.append(Suggestion(
                "todo_reminder", f"todo:{n.get('id')}:{due}", f"Urgent: '{text}' is due tomorrow. Set a reminder "
                "for 10 AM?", f"The high-priority to-do '{text}' is due tomorrow ({due}) with no reminder.",
                f"Remind me tomorrow at 10 AM to: {text}", 0.6, icon="📝"))
    return out


def rule_meeting_link(s: Snapshot) -> list[Suggestion]:
    """A meeting with a join link starts within 10 minutes (and the calendar reminder hasn't covered it)."""
    now = s.now.timestamp()
    out = []
    for e in s.events:
        if e.get("all_day") or not e.get("link"):
            continue
        lead = e["start"] - now
        mark = f"{e.get('uid')}@{int(e['start'])}"
        if 0 < lead <= 600 and mark not in s.reminded_events:
            mins = max(1, round(lead / 60))
            out.append(Suggestion(
                "meeting_link", f"meeting:{mark}", f"'{str(e.get('title', 'Meeting'))[:80]}' starts in {mins} min. "
                "Open the join link?", f"A calendar meeting starts in {mins} minutes and has a join link.",
                "Open the join link of my next meeting (calendar_events, then open_website).", 0.9,
                speak=f"Your next meeting starts in {mins} minutes and has a join link. Should I open it?",
                icon="📅"))
    return out


def _rate(log_: list[list[float]], since: float) -> float | None:
    """Change per hour over the samples after `since` (None with too little data)."""
    pts = [p for p in log_ if p[0] >= since]
    if len(pts) < 3 or pts[-1][0] - pts[0][0] < 20 * 60:
        return None
    return (pts[-1][1] - pts[0][1]) / ((pts[-1][0] - pts[0][0]) / 3600)


def battery_drain(s: Snapshot) -> float | None:
    """Percent per hour lost on battery over the last stretch unplugged (None = unknown)."""
    if not s.battery or s.battery[1]:
        return None
    start = s.now.timestamp() - 2 * 3600
    for t, _p, plugged in reversed([p if len(p) == 3 else [*p, 0] for p in s.battery_log]):
        if plugged:
            start = max(start, t + 1)
            break
    rate = _rate([[t, p] for t, p, *_ in s.battery_log], start)
    return -rate if rate is not None and rate < 0 else None


def rule_battery(s: Snapshot) -> list[Suggestion]:
    """On battery and low (or draining fast): offer power saver and lower brightness - never force it."""
    if not s.battery or s.battery[1]:
        return []
    percent = s.battery[0]
    offer = "Switch to the power saver plan and lower the brightness to 40 (ask me first)."
    if percent <= s.battery_saver_percent:
        return [Suggestion("battery", "battery_low", f"Battery at {percent:.0f}%. Switch to power saver and dim "
                           "the screen?", f"The PC is on battery at {percent:.0f}% (your threshold is "
                           f"{s.battery_saver_percent}%).", offer, 0.85, icon="🔋")]
    drain = battery_drain(s)
    if drain and drain >= 20 and percent / drain * 60 <= 75:
        mins = round(percent / drain * 60)
        return [Suggestion("battery", "battery_drain", f"Battery is draining fast (~{drain:.0f}%/hour, about "
                           f"{mins} min left). Save power?", f"Battery fell about {drain:.0f}% per hour recently.",
                           offer, 0.6, icon="🔋")]
    return []


def rule_disk(s: Snapshot) -> list[Suggestion]:
    """Low free space on C:, or free space falling fast."""
    free = s.disk_free_gb
    if free is None:
        return []
    action = "Check my disk space and tell me what I could clean up."
    if free < 10:
        return [Suggestion("disk", "disk_low", f"Only {free:.1f} GB free on C:. Look for things to clean up?",
                           f"C: has {free:.1f} GB free (below 10 GB).", action, 0.6 if free >= 5 else 0.8,
                           icon="💽")]
    rate = _rate(s.disk_log, s.now.timestamp() - 7 * 86400)
    if rate is not None and rate < 0 and s.disk_log and s.now.timestamp() - s.disk_log[0][0] >= 86400:
        days = (free - 5) / (-rate * 24)
        if days < 7:
            return [Suggestion("disk", "disk_trend", f"C: is filling up fast - at this rate it's nearly full in "
                               f"{max(1, round(days))} days. Check it?", f"Free space on C: fell about "
                               f"{-rate * 24:.1f} GB a day ({free:.0f} GB left).", action, 0.55, icon="💽")]
    return []


def rule_habit_time(s: Snapshot) -> list[Suggestion]:
    """A habit usually checked by now (learned from when it was checked before) isn't done today."""
    out = []
    minutes = s.now.hour * 60 + s.now.minute
    day = s.now.date()
    for h in s.habits:
        name = str(h.get("name", ""))
        times = s.habit_times.get(" ".join(name.lower().split()), [])
        if len(times) < 3 or (h.get("days", "daily") != "daily" and day.weekday() >= 5):
            continue
        if (h.get("log") or {}).get(day.isoformat(), 0) >= h.get("target", 1):
            continue
        usual = int(statistics.median(times))
        if usual + 30 <= minutes <= usual + 240:
            at = _clock(datetime.combine(day, datetime.min.time()) + timedelta(minutes=usual))
            out.append(Suggestion("habit_time", f"habit:{name}:{day}", f"You usually do '{name}' by {at}. "
                                  "Remind you in 15 minutes?", f"'{name}' was checked around {at} on "
                                  f"{len(times)} earlier days and isn't done today.",
                                  f"Remind me in 15 minutes to do: {name}", 0.55, icon="✅"))
    return out


def rule_night_pc(s: Snapshot) -> list[Suggestion]:
    """The PC is on late at night and nobody has used it for half an hour."""
    if not (s.now.hour >= 23 or s.now.hour < 5) or s.idle_seconds < 30 * 60:
        return []
    return [Suggestion("night_pc", f"night:{(s.now - timedelta(hours=6)).date()}", "It's late and the PC has "
                       f"been idle for {s.idle_seconds / 60:.0f} min. Put it to sleep?",
                       f"It's {_clock(s.now)} and there was no keyboard or mouse input for "
                       f"{s.idle_seconds / 60:.0f} minutes.", "Put the computer to sleep.", 0.5, icon="🌙",
                       quiet_ok=True)]


def _month_spent(items: list[dict], month: str, category: str | None) -> float:
    return sum(float(i.get("amount", 0)) for i in items
               if str(i.get("date", ""))[:7] == month and (category is None or i.get("category") == category))


def rule_budget(s: Snapshot) -> list[Suggestion]:
    """A monthly budget is nearly used up, or the spending pace will blow it before the month ends."""
    out = []
    day = s.now.date()
    month = day.isoformat()[:7]
    next_month = (day.replace(day=28) + timedelta(days=4)).replace(day=1)
    days_in_month = (next_month - day.replace(day=1)).days
    left_days = days_in_month - day.day
    for cat, limit in (s.budgets or {}).items():
        try:
            limit = float(limit)
        except (TypeError, ValueError):
            continue
        if limit <= 0:
            continue
        spent = _month_spent(s.expenses, month, None if cat == "total" else cat)
        label = "monthly budget" if cat == "total" else f"{cat} budget"
        share = spent / limit
        pace = spent / max(day.day, 1) * days_in_month
        if share >= 1:
            continue  # money.add_expense already says it is used up
        if share >= 0.85:
            text = f"You've used {share:.0%} of your {label} with {left_days} days left."
        elif share >= 0.5 and day.day >= 7 and pace > limit * 1.15:
            text = (f"At this pace you'll spend about {s.currency}{pace:,.0f} - over your {label} of "
                    f"{s.currency}{limit:,.0f}.")
        else:
            continue
        out.append(Suggestion("budget", f"budget:{cat}:{month}:{'near' if share >= 0.85 else 'pace'}",
                              text + " See where it went?", f"{s.currency}{spent:,.0f} of {s.currency}{limit:,.0f} "
                              f"spent in {month} ({share:.0%}).", "Show my expense summary for this month.",
                              0.6, icon="💸"))
    return out


def rule_workflow(s: Snapshot) -> list[Suggestion]:
    """The same few actions keep coming together: offer to save them as a routine."""
    existing = {tuple(" ".join(x.lower().split()) for x in steps) for steps in (s.routines or {}).values()}
    out = []
    for sig, info in (s.bursts or {}).items():
        steps = sig.split("|")
        if info.get("count", 0) < 3 or len(set(info.get("days", []))) < 2:
            continue
        if tuple(" ".join(x.lower().split()) for x in steps) in existing:
            continue
        shown = ", then ".join(steps)
        out.append(Suggestion("workflow", f"workflow:{sig}", f"You often do: {shown}. Save it as a routine "
                              "so one phrase does it all?", f"These {len(steps)} actions came together "
                              f"{info['count']} times on {len(set(info['days']))} different days.",
                              "Save a routine with these steps: " + "; ".join(steps)
                              + ". Ask me what to call it first.", 0.5, icon="🔁"))
    return out


def rule_birthday(s: Snapshot) -> list[Suggestion]:
    """A saved contact's birthday is tomorrow."""
    if s.now.hour < 9:
        return []
    tomorrow = s.now.date() + timedelta(days=1)
    mm_dd = f"{tomorrow.month:02d}-{tomorrow.day:02d}"
    out = []
    for name, bday in s.birthdays:
        if bday == mm_dd or (mm_dd == "02-28" and bday == "02-29" and tomorrow.year % 4):
            out.append(Suggestion("birthday", f"birthday:{name}:{tomorrow}", f"{name}'s birthday is tomorrow. "
                                  "Set a reminder to wish them?", f"{name}'s saved birthday is {bday}.",
                                  f"Remind me tomorrow at 9 AM to wish {name} a happy birthday.", 0.7, icon="🎂"))
    return out


def rule_messy_downloads(s: Snapshot) -> list[Suggestion]:
    """Watch mode: Downloads has many loose files."""
    if s.downloads_loose is None or s.downloads_loose < 25:
        return []
    year, week, _ = s.now.isocalendar()
    return [Suggestion("messy_downloads", f"downloads:{year}-W{week}", f"Downloads has {s.downloads_loose} loose "
                       "files. Sort them into folders? (You'll see the plan first.)",
                       f"Downloads has {s.downloads_loose} files at the top level (watch mode is on).",
                       "Organize my downloads folder - show me the plan first.", 0.45, icon="🗂")]


RULES = {
    "todo_reminder": ("To-dos due soon without a reminder", rule_todo_reminder),
    "meeting_link": ("Meeting about to start with a join link", rule_meeting_link),
    "battery": ("Low or fast-draining battery", rule_battery),
    "disk": ("Low or shrinking disk space", rule_disk),
    "habit_time": ("A habit usually done by now", rule_habit_time),
    "night_pc": ("PC left on late at night", rule_night_pc),
    "budget": ("Budget about to run out", rule_budget),
    "workflow": ("Repeated actions that could be a routine", rule_workflow),
    "birthday": ("Birthdays tomorrow", rule_birthday),
    "messy_downloads": ("Messy Downloads folder (watch mode)", rule_messy_downloads),
}


def candidates(s: Snapshot) -> list[Suggestion]:
    out = []
    for name, (_title, fn) in RULES.items():
        try:
            out.extend(fn(s))
        except Exception:  # noqa: BLE001 - one broken rule must not stop the others
            log.exception("proactive rule %s failed", name)
    return out


# -- state, scoring and limits --------------------------------------------------------------------
def _data(state) -> dict:
    d = state.get(KEY, {}) if state is not None else {}
    d = dict(d) if isinstance(d, dict) else {}
    for k, default in (("shown", []), ("muted", {}), ("dismissed", {}), ("habit_times", {}), ("bursts", {}),
                       ("battery_log", []), ("disk_log", []), ("burst", []), ("next_id", 1)):
        d.setdefault(k, default)
    return d


def _save(state, d: dict) -> None:
    state.set(KEY, d)


def score(sug: Suggestion, d: dict, now: float) -> float:
    """Base importance, lowered for rules the user often dismisses, raised a little for ones they accept."""
    recent = [x for x in d["shown"] if x.get("rule") == sug.rule and now - x.get("t", 0) < 14 * 86400]
    dismissed = sum(1 for x in recent if x.get("outcome") == "dismissed")
    accepted = sum(1 for x in recent if x.get("outcome") == "accepted")
    return max(0.0, min(1.0, sug.score * (0.75 ** dismissed) * (1.1 ** min(accepted, 3))))


def in_quiet_hours(hour: int, quiet: str) -> bool:
    try:
        a, b = (int(x) % 24 for x in str(quiet).split("-"))
    except (ValueError, TypeError):
        return False
    if a == b:
        return False
    return a <= hour < b if a < b else hour >= a or hour < b


def pick(state, settings, snap: Snapshot, busy: bool = False) -> dict | None:
    """The suggestion to show now (recorded as shown), or None. `busy` = focus or practice mode."""
    if not getattr(settings, "proactive_suggestions", True) or busy or state is None:
        return None
    now = snap.now.timestamp()
    d = _data(state)
    shown = d["shown"]
    if shown and now - shown[-1].get("t", 0) < MIN_GAP:
        return None
    per_hour = max(0, int(getattr(settings, "suggestions_per_hour", 2) or 0))
    if sum(1 for x in shown if now - x.get("t", 0) < 3600) >= per_hour:
        return None
    quiet = in_quiet_hours(snap.now.hour, getattr(settings, "quiet_hours", ""))
    best, best_score = None, THRESHOLD
    for sug in candidates(snap):
        if sug.rule in d["muted"] or (quiet and not sug.quiet_ok):
            continue
        if now - d["dismissed"].get(sug.key, -1e12) < DISMISS_COOLDOWN:
            continue
        if any(x.get("key") == sug.key and now - x.get("t", 0) < KEY_COOLDOWN for x in shown):
            continue
        sc = score(sug, d, now)
        if sc >= best_score + (1e-9 if best else 0):
            best, best_score = sug, sc
    if best is None:
        return None
    entry = {**asdict(best), "id": d["next_id"], "t": now, "score": round(best_score, 3), "outcome": None,
             "speak_ok": not quiet}
    d["next_id"] += 1
    d["shown"] = (shown + [entry])[-MAX_SHOWN:]
    _save(state, d)
    return entry


def _find(d: dict, sid: int = 0) -> dict | None:
    if not d["shown"]:
        return None
    if not sid:
        return d["shown"][-1]
    return next((x for x in d["shown"] if x.get("id") == int(sid)), None)


def feedback(state, sid: int, outcome: str, now: float | None = None) -> dict | None:
    """Accepted / dismissed / not_useful for suggestion `sid` (0 = the latest). Not useful mutes its rule."""
    if outcome not in OUTCOMES:
        raise ValueError(outcome)
    now = time.time() if now is None else now
    d = _data(state)
    entry = _find(d, sid)
    if entry is None:
        return None
    entry["outcome"] = outcome
    if outcome in ("dismissed", "accepted"):
        d["dismissed"][entry["key"]] = now
    if outcome == "not_useful":
        d["muted"][entry["rule"]] = now
        d["dismissed"][entry["key"]] = now
    d["dismissed"] = {k: t for k, t in d["dismissed"].items() if now - t < 7 * 86400}
    _save(state, d)
    return entry


def explain(state, sid: int = 0) -> dict | None:
    return _find(_data(state), sid)


def set_muted(state, rule: str, muted: bool) -> None:
    d = _data(state)
    if muted:
        d["muted"][rule] = time.time()
    else:
        d["muted"].pop(rule, None)
    _save(state, d)


def muted_rules(state) -> list[str]:
    return sorted(_data(state)["muted"])


# -- learning ----------------------------------------------------------------------------------------
def _step(name: str, args: dict) -> str | None:
    """A readable step for a routine, from a tool call the user asked for."""
    a = args or {}
    text = {
        "open_app": lambda: f"open {a.get('name', '')}",
        "open_website": lambda: f"open {a.get('url', '')}",
        "play_youtube": lambda: f"play {a.get('query', '')} on YouTube",
        "volume": lambda: f"volume {a.get('action', '')}",
        "media_control": lambda: f"media {a.get('action', '')}",
        "dark_mode": lambda: f"dark mode {a.get('mode', '')}",
        "focus_mode": lambda: f"focus mode {a.get('minutes', 25)} minutes",
        "game_launcher": lambda: f"open {a.get('launcher', '')}",
        "brightness": lambda: f"brightness {a.get('level', '')}",
        "power_mode": lambda: f"power mode {a.get('mode', '')}",
    }.get(name)
    if text is None:
        return None
    out = " ".join(str(text()).split())[:60]
    return out.replace("|", " ") or None


def _close_burst(d: dict, day: str) -> None:
    steps = [s for _t, s in d["burst"]]
    d["burst"] = []
    if not 2 <= len(steps) <= 5 or len(set(steps)) != len(steps):
        return
    sig = "|".join(steps)
    info = d["bursts"].get(sig, {"count": 0, "days": []})
    info["count"] += 1
    info["days"] = (info["days"] + [day])[-10:]
    d["bursts"][sig] = info
    if len(d["bursts"]) > 40:  # keep the most frequent
        d["bursts"] = dict(sorted(d["bursts"].items(), key=lambda kv: -kv[1]["count"])[:30])


def observe(state, name: str, args: dict, result: dict, now: float | None = None) -> None:
    """Learn from a tool the user ran: when habits get checked, and which actions come together."""
    if state is None or not isinstance(result, dict) or "error" in result:
        return
    if str(result.get("result", "")).startswith("Cancelled"):
        return
    now = time.time() if now is None else now
    dt = datetime.fromtimestamp(now)
    d = _data(state)
    changed = False
    if d["burst"] and now - d["burst"][-1][0] > BURST_GAP:
        _close_burst(d, datetime.fromtimestamp(d["burst"][-1][0]).date().isoformat())
        changed = True
    if name == "check_habit" and not (args or {}).get("when"):
        key = " ".join(str((args or {}).get("name", "")).lower().split())
        if key:
            d["habit_times"][key] = (d["habit_times"].get(key, []) + [dt.hour * 60 + dt.minute])[-14:]
            changed = True
    if name in WORKFLOW_TOOLS:
        step = _step(name, args)
        if step:
            d["burst"] = (d["burst"] + [[now, step]])[-6:]
            changed = True
    if changed:
        _save(state, d)


def flush(state, now: float | None = None) -> None:
    """Close a sequence that has gone quiet (called by the background loop)."""
    now = time.time() if now is None else now
    d = _data(state)
    if d["burst"] and now - d["burst"][-1][0] > BURST_GAP:
        _close_burst(d, datetime.fromtimestamp(d["burst"][-1][0]).date().isoformat())
        _save(state, d)


def record_readings(state, battery: tuple[float, bool] | None, disk_free_gb: float | None,
                    now: float | None = None) -> None:
    """Keep a short history of battery and disk readings for the trend rules."""
    now = time.time() if now is None else now
    d = _data(state)
    changed = False
    if battery is not None and (not d["battery_log"] or now - d["battery_log"][-1][0] >= READING_EVERY):
        d["battery_log"] = (d["battery_log"] + [[now, float(battery[0]), 1 if battery[1] else 0]])[-MAX_READINGS:]
        changed = True
    if disk_free_gb is not None and (not d["disk_log"] or now - d["disk_log"][-1][0] >= 6 * 3600):
        d["disk_log"] = (d["disk_log"] + [[now, round(float(disk_free_gb), 2)]])[-60:]
        changed = True
    if changed:
        _save(state, d)


# -- reading the real PC --------------------------------------------------------------------------------
def _read_battery():
    import psutil

    b = psutil.sensors_battery()
    return (float(b.percent), bool(b.power_plugged)) if b else None


def _read_disk() -> float | None:
    import psutil

    return psutil.disk_usage("C:\\" if os.name == "nt" else "/").free / 1024 ** 3


def loose_files(folder) -> int:
    from pathlib import Path

    try:
        return sum(1 for p in Path(folder).iterdir() if p.is_file() and not p.name.startswith("."))
    except OSError:
        return 0


def gather(ctx, now: datetime | None = None) -> Snapshot:
    """A snapshot of the running app and the PC. Every source is optional: a failing one is just left empty."""
    now = now or datetime.now()
    settings, state = ctx.settings, ctx.state
    snap = Snapshot(now=now, battery_saver_percent=int(getattr(settings, "battery_saver_percent", 25) or 0),
                    currency=getattr(settings, "currency", "₹") or "₹")

    def safe(fn):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            return None

    if state is not None:
        snap.todos = [n for n in (state.get("notes", []) or []) if n.get("todo") and not n.get("done")]
        snap.reminders = list(state.get("reminders", []) or [])
        snap.reminded_events = set(state.get("calendar_reminded", []) or [])
        snap.routines = dict(state.get("routines", {}) or {})
    from jarvis.tools import calendar_sync

    snap.events = safe(lambda: calendar_sync.events(settings)) or []
    snap.battery = safe(_read_battery)
    snap.disk_free_gb = safe(_read_disk)
    if state is not None:
        record_readings(state, snap.battery, snap.disk_free_gb, now.timestamp())
        d = _data(state)
        snap.battery_log, snap.disk_log = d["battery_log"], d["disk_log"]
        snap.habit_times, snap.bursts = d["habit_times"], d["bursts"]

    def habits_list():
        from jarvis.tools.habits import habits

        return habits().load().get("habits", [])

    snap.habits = safe(habits_list) or []

    def money():
        from jarvis.tools.money import expenses

        return expenses().load()

    data = safe(money) or {}
    snap.expenses, snap.budgets = data.get("items", []), data.get("budgets", {})

    def birthdays():
        from jarvis.tools.contacts import all_contacts

        return [(p["name"], p["birthday"]) for p in all_contacts(ctx.memory).values() if p.get("birthday")]

    snap.birthdays = safe(birthdays) or []

    def idle():
        from jarvis.idle import seconds_idle

        return seconds_idle()

    snap.idle_seconds = safe(idle) or 0.0
    if getattr(settings, "organize_watch", False):
        from jarvis.tools.files import SHORTCUTS

        snap.downloads_loose = loose_files(SHORTCUTS["downloads"])
    return snap


def spoken(entry: dict) -> str:
    """The instruction for the live model to offer a suggestion out loud (trusted text only)."""
    line = entry.get("speak") or entry.get("title", "")
    return ("(system: proactive suggestion) In one short sentence, offer this to the user: " + line
            + " A card with Accept / Dismiss is on screen. Don't do it unless they say yes; if they say it's not "
            "useful, call suggestion_feedback with not_useful.")

