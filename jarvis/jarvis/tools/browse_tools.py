"""Voice tools for the careful web researcher (see ``jarvis.agents.browser``)."""

from __future__ import annotations

from jarvis.tools import tool

DEPTHS = {"quick": 1, "normal": 2, "deep": 3}


def _depth(depth) -> int:
    if isinstance(depth, str) and depth.lower() in DEPTHS:
        return DEPTHS[depth.lower()]
    try:
        return max(1, min(int(float(depth)), 3))
    except (TypeError, ValueError):
        return 1


@tool(
    "browse_and_report",
    "Research a topic (or start from one web page) by reading several pages, then write a markdown report with "
    "sources, saved in Documents/Jarvis Reports after the user agrees. depth 1 = quick (3 pages), 2 = normal, "
    "3 = deep (8 pages, follows relevant links). Read-only: never logs in, fills forms or buys anything. Say "
    "the short summary; a card shows the sources.",
    {"topic": {"type": "string", "description": "What to research"},
     "url": {"type": "string", "description": "Optional page to start from"},
     "depth": {"type": "integer", "description": "1, 2 or 3 (default 1)"}},
    required=["topic"],
    slow=True,
)
def browse_and_report(ctx, topic: str, url: str = "", depth: int = 1):
    from jarvis.agents import browser, team

    topic = " ".join(str(topic or url).split())[:200]
    if not topic:
        return {"error": "Tell me what to research."}
    url = (url or "").strip()
    if url and "://" not in url:
        url = f"https://{url}"
    if url and not browser.public_url(url):
        return {"error": "I only read public http(s) web pages."}
    run = team.current()
    stop = (lambda: run.cancel_event.is_set()) if run is not None and run.active else (lambda: False)
    r = browser.research(topic, ctx.llm, _depth(depth), url or None, should_stop=stop)
    if not r.pages:
        why = "; ".join(f"{u}: {w}" for u, w in r.skipped[:3])
        return {"error": f"I couldn't read any pages about that. {why}".strip()}
    saved = None
    if ctx.confirm.confirm(f"Save the report on '{topic[:60]}' to Documents/Jarvis Reports?"):
        saved = browser.save_report(r)
    results = [{"title": p.title or p.url, "url": p.url,
                "snippet": next((f for f, s in r.facts if s == i), "")} for i, p in enumerate(r.pages, 1)]
    out = {"query": topic, "mode": "report", "results": results, "facts": [f for f, _ in r.facts][:20],
           "summary": r.summary,
           "result": r.summary + (f" Report saved: {saved.name}." if saved else " The report wasn't saved.")}
    if saved:
        out["path"] = str(saved)
    if any(p.suspicious for p in r.pages):
        out["note"] = "Some pages contained text that tried to give an AI instructions; it was ignored."
    return out


@tool(
    "read_web_page",
    "Read one public web page (readable text, title and links). Read-only; respects robots.txt.",
    {"url": {"type": "string"}},
    required=["url"],
    slow=True,
)
def read_web_page(ctx, url: str):
    from jarvis.agents import browser

    url = url.strip()
    if "://" not in url:
        url = f"https://{url}"
    try:
        page = browser.Fetcher().fetch(url)
    except browser.FetchError as exc:
        return {"error": f"Couldn't read {url}: {exc}"}
    links = [{"text": t, "url": u} for u, t in page.links if t and browser.public_url(u)][:15]
    return {"url": page.url, "title": page.title, "text": page.text[:8000], "links": links}
