"""Search past conversations: "what did I ask yesterday?", "what did we say about the trip?"."""

from __future__ import annotations

from datetime import datetime, timedelta

from jarvis.tools import tool

STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "is", "was", "what", "did", "we", "i", "me", "my",
        "about", "kya", "ka", "ki", "ke", "tha", "thi", "hai", "maine", "humne", "baat", "kaha", "poocha"}


@tool(
    "search_chat_history",
    "Search earlier conversations with the user (saved on this computer). Use for 'what did I ask "
    "yesterday', 'humne trip ke baare mein kya baat ki thi'. Give keywords; days limits how far back.",
    {"query": {"type": "string", "description": "keywords; empty = just the most recent messages"},
     "days": {"type": "integer", "description": "only the last N days (optional)"}},
)
def search_chat_history(ctx, query: str = "", days: int | None = None):
    from jarvis.config import home_dir
    from jarvis.history import ChatHistory

    items = ChatHistory(home_dir() / "history.jsonl").recent(1000)
    if days:
        since = (datetime.now() - timedelta(days=int(days))).timestamp()
        items = [i for i in items if i.get("t", 0) >= since]
    words = [w for w in query.lower().split() if w not in STOP]
    if words:
        scored = []
        for i, item in enumerate(items):
            text = item["text"].lower()
            score = sum(1 for w in words if w in text)
            if score:
                scored.append((score, i))
        keep = sorted(i for _, i in sorted(scored, reverse=True)[:8])
        # include the reply after each hit, so the answer comes with the question
        picked = sorted({j for i in keep for j in (i, i + 1) if j < len(items)})
        items = [items[j] for j in picked]
    else:
        items = items[-10:]
    if not items:
        return "Nothing found in earlier conversations."
    rows = [{"when": datetime.fromtimestamp(i.get("t", 0)).strftime("%d %b %I:%M %p"),
             "who": i["role"], "text": i["text"][:300]} for i in items]
    return {"matches": rows}
