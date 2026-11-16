"""AI meeting assistant: record a meeting, transcribe it, and write notes, action items and a follow-up e-mail.

Recording starts only when the user asks (`start_meeting`) and shows a red REC pill. The "meetings"
permission in Settings > Privacy can turn it off completely. The audio and notes stay in
~/.jarvis/meetings/ until the user deletes them (here or in Settings > Privacy).
"""

from __future__ import annotations

import time

from jarvis import meetings
from jarvis.tools import tool

NOT_FOUND = "I couldn't find that meeting. Say list_meetings to see the saved ones."


def _card(info: dict, mid: str) -> dict:
    """What the meeting card in the window shows."""
    notes = info.get("notes") or {}
    return {"id": mid, "title": info.get("title", ""), "when": meetings.when_text(info),
            "status": info.get("status", ""), "sources": info.get("sources", []), **notes}


def _after_notes(info: dict) -> str:
    n = info.get("notes") or {}
    parts = ["Give a 2-3 sentence summary out loud (the card shows the rest)."]
    if n.get("action_items"):
        parts.append(f"Offer to add the {len(n['action_items'])} action item(s) to the to-do list "
                     "(save_meeting_actions) and to set reminders for the ones with a due date (set_reminder).")
    if (n.get("email") or {}).get("body"):
        parts.append("Offer to open the follow-up e-mail as a draft with draft_email (ask who it goes to; "
                     "it is only a draft - the user sends it).")
    return " ".join(parts)


@tool(
    "start_meeting",
    "Record a meeting and take notes: the microphone plus the PC's sound when possible. ONLY when the user "
    "explicitly asks to record / take notes of a meeting or call - never on your own, never because a calendar "
    "meeting is starting (ask first). 'title' is a short name for the meeting.",
    {"title": {"type": "string"}},
)
def start_meeting(ctx, title: str = "Meeting"):
    running = meetings.current()
    if running is not None:
        return {"error": f"Already recording '{running.meeting.title}'. Say stop_meeting first."}
    try:
        session = meetings.begin(ctx, title or "Meeting")
    except Exception as exc:
        return {"error": f"Couldn't start recording: {type(exc).__name__}: {exc}"}
    m = session.meeting
    if ctx.events is not None:
        ctx.events.emit("log", text=f"🔴 Recording started: {m.title} ({' + '.join(m.sources)})")
    reply = (f"Recording '{m.title}' from: {', '.join(m.sources)}. Say 'recording started' ONCE in one short line"
             + (f" and mention briefly: {m.note}" if m.note else "") +
             ". Then stay quiet during the meeting unless the user speaks to you by name.")
    return {"result": reply, "id": m.id, "sources": m.sources}


@tool("stop_meeting", "Stop the meeting recording and make the notes: summary, decisions, action items with owners "
                      "and due dates, open questions and a follow-up e-mail draft.", slow=True)
def stop_meeting(ctx):
    session = meetings.end(ctx)
    if session is None:
        return {"error": "No meeting is being recorded."}
    try:
        info = meetings.make_notes(ctx, session.folder)
    except Exception as exc:
        if ctx.events is not None:
            ctx.events.emit("meeting", state="stopped", id=session.meeting.id, title=session.meeting.title)
        return {"error": f"Recording saved, but the notes failed ({type(exc).__name__}: {exc}). "
                         "Try meeting_notes later."}
    if ctx.events is not None:
        ctx.events.emit("meeting", state="stopped", id=session.meeting.id, title=session.meeting.title)
    return {"result": "Recording stopped and notes made. " + _after_notes(info),
            "meeting_notes": _card(info, session.meeting.id)}


@tool("meeting_status", "Is a meeting being recorded right now, for how long, and from which sources.")
def meeting_status(ctx):
    s = meetings.current()
    if s is None:
        return {"recording": False, "result": "No meeting is being recorded."}
    return {"recording": True, "title": s.meeting.title, "minutes": round(s.recorder.seconds / 60, 1),
            "sources": s.meeting.sources, "chunks_transcribed": len(s.info.get("chunks", [])),
            "since": time.strftime("%I:%M %p", time.localtime(s.meeting.started)).lstrip("0")}


@tool("list_meetings", "The saved meeting recordings, newest first (id, title, date, length, notes ready).")
def list_meetings(ctx, limit: int = 10):
    rows = meetings.all_meetings()[: max(1, min(int(limit or 10), 50))]
    if not rows:
        return {"result": "No meetings recorded yet."}
    return {"meetings": [{"id": r["id"], "title": r.get("title", ""), "when": meetings.when_text(r),
                          "status": r.get("status", ""), "notes": bool(r.get("notes"))} for r in rows]}


@tool("meeting_notes", "Show the notes of a recorded meeting (summary, decisions, action items, open questions, "
                       "e-mail draft). id: from list_meetings, part of the title, or 'last'.",
      {"id": {"type": "string"}}, slow=True)
def meeting_notes(ctx, id: str = "last"):  # noqa: A002
    folder = meetings.find(id)
    if folder is None:
        return {"error": NOT_FOUND}
    s = meetings.current()
    info = meetings.load(folder)
    if s is not None and s.folder == folder:
        return {"result": "This meeting is still being recorded; notes come when it stops.",
                "transcribed_so_far": meetings.transcript(info)[-3000:]}
    if not info.get("notes"):
        info = meetings.make_notes(ctx, folder)
    return {"result": _after_notes(info), "meeting_notes": _card(info, folder.name)}


@tool("ask_meeting", "Answer a question about a recorded meeting ('what did Priya say about the budget?'). "
                     "id: from list_meetings, part of the title, or 'last'.",
      {"id": {"type": "string"}, "question": {"type": "string"}}, required=["question"], slow=True)
def ask_meeting(ctx, question: str, id: str = "last"):  # noqa: A002
    folder = meetings.find(id)
    if folder is None:
        return {"error": NOT_FOUND}
    return {"meeting": meetings.load(folder).get("title", ""), "answer": meetings.ask(ctx, folder, question)}


@tool("save_meeting_actions", "Add a meeting's action items to the to-do list (with their due dates). Only after "
                              "the user said yes. id: from list_meetings or 'last'; only_mine=true keeps just the "
                              "items owned by the user.",
      {"id": {"type": "string"}, "only_mine": {"type": "boolean"}})
def save_meeting_actions(ctx, id: str = "last", only_mine: bool = False):  # noqa: A002
    from jarvis.guardrails import scan
    from jarvis.tools.notes import add_note

    folder = meetings.find(id)
    if folder is None:
        return {"error": NOT_FOUND}
    items = (meetings.load(folder).get("notes") or {}).get("action_items") or []
    me = (ctx.settings.user_name or "").strip().lower()
    added = []
    for a in items:
        owner = a.get("owner", "")
        if only_mine and not (owner.lower() in {"me", "i", "you"} or (me and me in owner.lower())):
            continue
        text = a["task"] + (f" ({owner})" if owner and not only_mine else "")
        if scan(text):
            continue  # never save a secret
        reply = add_note(ctx, text, todo=True, due=a.get("due", ""), priority="normal")
        if isinstance(reply, str):
            added.append(text)
    if not added:
        return {"result": "There were no action items to add."}
    return {"result": f"Added {len(added)} to-do(s) from the meeting.", "added": added}


@tool("delete_meeting", "Delete a recorded meeting (audio, transcript and notes) for good. The user confirms on "
                        "screen. id: from list_meetings, part of the title, or 'last'.",
      {"id": {"type": "string"}}, required=["id"])
def delete_meeting(ctx, id: str):  # noqa: A002
    folder = meetings.find(id)
    if folder is None:
        return {"error": NOT_FOUND}
    s = meetings.current()
    if s is not None and s.folder == folder:
        return {"error": "That meeting is still being recorded - stop it first."}
    title = meetings.load(folder).get("title", folder.name)
    if not ctx.confirm.confirm(f"Delete the meeting '{title}' (audio, transcript and notes)? This can't be undone."):
        return {"result": "Cancelled - nothing was deleted."}
    meetings.delete(folder)
    if ctx.events is not None:
        ctx.events.emit("meeting", state="deleted", id=folder.name, title=title)
    return {"result": f"Deleted the meeting '{title}'."}
