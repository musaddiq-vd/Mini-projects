"""'Aaj ka recap' - a summary of the user's day with Jarvis, and a nudge to plan tomorrow."""

from __future__ import annotations

from datetime import datetime, time as dtime

from jarvis.tools import tool


def _today_start() -> float:
    return datetime.combine(datetime.now().date(), dtime.min).timestamp()


def day_facts(ctx) -> dict:
    from jarvis.config import home_dir
    from jarvis.history import ChatHistory

    start = _today_start()
    st = ctx.state
    history = [i for i in ChatHistory(home_dir() / "history.jsonl").recent(2000) if i.get("t", 0) >= start]
    asked = [i["text"][:80] for i in history if i["role"] == "user"]
    notes = (st.get("notes", []) if st is not None else []) or []
    done_today = [n["text"] for n in notes if n.get("done") and n.get("done_at", 0) >= start]
    still_open = [n["text"] for n in notes if n.get("todo") and not n.get("done")]
    now = datetime.now().timestamp()
    reminders = (st.get("reminders", []) if st is not None else []) or []
    tomorrow_end = start + 2 * 86400
    upcoming = [f"{datetime.fromtimestamp(r['due']):%a %I:%M %p} - {r['message']}" for r in
                sorted(reminders, key=lambda r: r["due"]) if now < r["due"] < tomorrow_end]
    ach = (st.get("achievements", {}) if st is not None else {}) or {}
    badges = [k for k, t in (ach.get("unlocked") or {}).items() if t >= start]
    facts = {
        "date": datetime.now().strftime("%A %d %B"),
        "messages_today": len(history),
        "things_you_asked": asked[-15:],
        "todos_done_today": done_today,
        "todos_still_open": still_open[:10],
        "reminders_coming_up": upcoming[:6],
        "badges_unlocked_today": badges,
        "streak_days": (ach.get("counts") or {}).get("streak_now", 0),
    }
    if (home_dir() / "english.json").exists():  # only if practice mode has been used
        try:
            from jarvis.english.store import EnglishStore

            store = EnglishStore(home_dir() / "english.json")
            summary = store.summary()
            facts["language_practice"] = {"done_today": summary["done"], "of": summary["total"],
                                          "streak": summary["streak"], "xp": summary["xp"]}
        except Exception:
            pass
    return facts


@tool(
    "daily_recap",
    "A summary of the user's day with you: what they asked, to-dos finished and still open, upcoming reminders, "
    "badges, practice. Use for 'aaj ka recap', 'how was my day', 'what did I do today', and for the evening recap.",
)
def daily_recap(ctx):
    facts = day_facts(ctx)
    facts["how_to_answer"] = (
        "Give a warm, upbeat recap in 4-6 short spoken sentences in the user's language: what they got done "
        "(celebrate it), what's still open, what's coming up. Then ask what they want to do tomorrow and offer to "
        "add it to their list or set reminders. Don't list everything; pick the highlights.")
    return facts
