"""Voice control for proactive suggestions: turn them on or off, limit them, mute a kind, and ask why one came up.

The engine itself is jarvis.proactive; the background services call it every minute.
"""

from __future__ import annotations

from datetime import datetime

from jarvis import proactive
from jarvis.tools import tool

ACTIONS = ["status", "on", "off", "per_hour", "quiet_hours", "mute", "unmute", "watch_downloads_on",
           "watch_downloads_off"]


def _save(ctx) -> str | None:
    from jarvis.config import save_settings

    try:
        save_settings(ctx.settings)
    except OSError as exc:
        return f"Couldn't save the setting: {exc}"
    return None


def _rule(name: str) -> str | None:
    key = "_".join(str(name).lower().replace("-", " ").split())
    if key in proactive.RULES:
        return key
    return next((r for r, (title, _) in proactive.RULES.items() if key and key.replace("_", " ") in title.lower()),
                None)


@tool(
    "proactive_settings",
    "Jarvis's proactive suggestions (offers of help before being asked: to-dos without a reminder, meetings with a "
    "join link, low battery or disk, habits, budgets, birthdays, repeated actions, PC left on at night). "
    "action: status | on | off | per_hour (value = max suggestions per hour) | quiet_hours (value like '22-8', "
    "'' = none) | mute / unmute (value = a kind from status) | watch_downloads_on / watch_downloads_off "
    "(suggest organizing Downloads when it gets messy).",
    {"action": {"type": "string", "enum": ACTIONS}, "value": {"type": "string"}},
)
def proactive_settings(ctx, action: str = "status", value: str = ""):
    s = ctx.settings
    action = (action or "status").strip().lower()
    if action == "status":
        muted = proactive.muted_rules(ctx.state) if ctx.state is not None else []
        return {"enabled": s.proactive_suggestions, "per_hour": s.suggestions_per_hour,
                "quiet_hours": s.quiet_hours or "none", "watch_downloads": s.organize_watch,
                "kinds": {r: title + (" (muted)" if r in muted else "") for r, (title, _) in proactive.RULES.items()}}
    if action in ("on", "off"):
        s.proactive_suggestions = action == "on"
        msg = "Proactive suggestions are on." if action == "on" else "Okay, no more proactive suggestions."
    elif action == "per_hour":
        try:
            n = int(float(value))
        except (TypeError, ValueError):
            return {"error": "Say a number, like 'at most 1 per hour'."}
        s.suggestions_per_hour = max(0, min(n, 6))
        msg = f"At most {s.suggestions_per_hour} suggestion(s) an hour."
    elif action == "quiet_hours":
        text = str(value).replace(" ", "").replace("to", "-")
        try:
            if text:
                a, b = (int(x) for x in text.split("-"))
                if not (0 <= a < 24 and 0 <= b < 24):
                    raise ValueError
        except ValueError:
            return {"error": "Give hours like '22-8' (10 pm to 8 am)."}
        s.quiet_hours = text
        msg = f"Quiet hours: {text}." if text else "No quiet hours."
    elif action in ("mute", "unmute"):
        if ctx.state is None:
            return {"error": "This needs the full Jarvis app."}
        rule = _rule(value)
        if rule is None:
            return {"error": f"Unknown kind '{value}'. Kinds: {', '.join(proactive.RULES)}."}
        proactive.set_muted(ctx.state, rule, action == "mute")
        return f"{'Muted' if action == 'mute' else 'Unmuted'}: {proactive.RULES[rule][0].lower()}."
    elif action in ("watch_downloads_on", "watch_downloads_off"):
        s.organize_watch = action.endswith("_on")
        msg = ("I'll suggest organizing Downloads when it gets messy." if s.organize_watch
               else "Stopped watching Downloads.")
    else:
        return {"error": f"Unknown action '{action}'."}
    error = _save(ctx)
    return {"error": error} if error else msg


@tool(
    "why_did_you_suggest",
    "Explain why Jarvis made a proactive suggestion (the latest one, or by its number).",
    {"id": {"type": "integer", "description": "0 = the latest"}},
)
def why_did_you_suggest(ctx, id: int = 0):  # noqa: A002 - matches the tool argument name
    entry = proactive.explain(ctx.state, int(id or 0)) if ctx.state is not None else None
    if entry is None:
        return "I haven't made any suggestions yet."
    out = {"id": entry["id"], "kind": proactive.RULES.get(entry["rule"], (entry["rule"],))[0],
           "why": entry.get("why", ""), "when": datetime.fromtimestamp(entry.get("t", 0)).strftime("%d %b %I:%M %p"),
           "answer": entry.get("outcome") or "no answer yet",
           "tip": "If it wasn't useful, the user can say so: suggestion_feedback not_useful mutes this kind."}
    if entry["rule"] != "meeting_link":  # a meeting title comes from the calendar: not repeated to the model
        out["suggestion"] = entry.get("title", "")
    return out


@tool(
    "suggestion_feedback",
    "The user's answer to a proactive suggestion: accept (then do what it offered), dismiss (not now), or "
    "not_useful (mutes that kind of suggestion).",
    {"answer": {"type": "string", "enum": list(proactive.OUTCOMES) + ["accept", "dismiss"]},
     "id": {"type": "integer", "description": "0 = the latest"}},
    required=["answer"],
)
def suggestion_feedback(ctx, answer: str, id: int = 0):  # noqa: A002
    if ctx.state is None:
        return {"error": "This needs the full Jarvis app."}
    outcome = {"accept": "accepted", "dismiss": "dismissed"}.get(answer, answer)
    if outcome not in proactive.OUTCOMES:
        return {"error": f"Unknown answer '{answer}'."}
    entry = proactive.feedback(ctx.state, int(id or 0), outcome)
    if entry is None:
        return {"error": "There's no suggestion to answer."}
    if ctx.events is not None:
        ctx.events.emit("suggestion_answered", id=entry["id"], outcome=outcome)
    if outcome == "accepted":
        return {"result": "Accepted. Now do it.", "do": entry.get("action", "")}
    if outcome == "not_useful":
        kind = proactive.RULES.get(entry["rule"], (entry["rule"],))[0].lower()
        return f"Got it - I won't suggest '{kind}' again. (Unmute it any time with proactive_settings.)"
    return "Okay, dismissed."
