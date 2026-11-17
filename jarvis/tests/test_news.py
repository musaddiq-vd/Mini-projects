"""News: the user's own country, fresh top headlines from newsroom feeds, no scrambled snippets, retries."""

import sys
import types

import pytest

from jarvis.tools import web

FEED = b"""<?xml version="1.0"?><rss><channel><title>Some Feed | Junk</title>
<item><title>Older story</title><link>https://a.example/1</link><pubDate>Tue, 17 Nov 2026 05:00:00 +0530</pubDate>
<description>&lt;p&gt;An older &lt;b&gt;story&lt;/b&gt; text&lt;/p&gt;</description></item>
<item><title>Newest story</title><link>https://a.example/2</link><pubDate>Tue, 17 Nov 2026 07:30:00 +0530</pubDate>
<description>Fresh news from today</description></item>
<item><title>Third story</title><link>https://a.example/3</link><pubDate>Tue, 17 Nov 2026 06:00:00 +0530</pubDate>
<description>6H 7:I 7@C @=5 t==:4@EE rC66 AC@3=6&gt;i</description></item>
</channel></rss>"""


class FakeResponse:
    def __init__(self, body=FEED, status=200):
        self.status_code = status
        self.raw = types.SimpleNamespace(read=lambda n, decode_content=True: body[:n])

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_region_from_time_zone(monkeypatch):
    import time

    monkeypatch.setattr(time, "tzname", ("India Standard Time", "India Daylight Time"))
    assert web.home_region() == "in-en"
    monkeypatch.setattr(time, "tzname", ("Pacific Standard Time", ""))
    assert web.home_region() == "us-en"
    monkeypatch.setattr(time, "tzname", ("Somewhere Odd", ""))
    assert web.home_region() == "wt-wt"


def test_scrambled_snippets_are_dropped():
    assert web.readable("6H 7:I 7@C @=5 t==:4@EE rC66 AC@3=6>i 'p? xD=2?5") == ""
    assert web.readable("Check out the latest news for Nov. 15, 2026.") != ""
    assert web.readable(None) == "" and web.readable("   ") == ""


@pytest.mark.parametrize("query, generic", [("top news headlines", True), ("today's top headlines", True),
                                             ("aaj ki khabar", True), ("latest news", True),
                                             ("news about cricket", False), ("tesla stock news", False)])
def test_headline_queries(query, generic):
    assert web.is_headlines_query(query) is generic


def test_top_headlines_newest_first_from_several_newsrooms(monkeypatch):
    monkeypatch.setitem(web.HEADLINE_FEEDS, "in-en", ["https://www.thehindu.com/x.rss", "https://ndtv.example/y"])
    monkeypatch.setattr(web.requests, "get", lambda url, **kw: FakeResponse(
        FEED if "thehindu" in url else FEED.replace(b"story", b"report").replace(b"07:30", b"07:00")))
    items = web.top_headlines("in-en")
    assert items[0]["title"] == "Newest story" and items[0]["source"] == "The Hindu"
    assert {x["source"] for x in items} == {"The Hindu", "NDTV"}
    assert all(sum(1 for x in items if x["source"] == s) <= 2 for s in ("The Hindu", "NDTV"))
    assert [x for x in items if x["title"] == "Third story"][0]["snippet"] == ""  # scrambled text dropped
    assert "<" not in items[-1]["snippet"]


def test_a_broken_or_huge_feed_is_skipped(monkeypatch):
    monkeypatch.setitem(web.HEADLINE_FEEDS, "in-en", ["https://bad.example", "https://www.thehindu.com/ok.rss"])

    def get(url, **kw):
        if "bad" in url:
            return FakeResponse(b"x" * (web.MAX_FEED_BYTES + 10))
        return FakeResponse()

    monkeypatch.setattr(web.requests, "get", get)
    assert {x["source"] for x in web.top_headlines("in-en")} == {"The Hindu"}


def _fake_ddgs(monkeypatch, answers):
    calls = []

    class DDGS:
        def news(self, query, **kw):
            calls.append(("news", query, kw))
            ans = answers.pop(0) if answers else []
            if isinstance(ans, Exception):
                raise ans
            return ans

        text = news

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=DDGS))
    return calls


def test_headlines_use_the_feeds(registry, ctx, monkeypatch):
    monkeypatch.setattr(web, "home_region", lambda: "in-en")
    monkeypatch.setattr(web, "top_headlines", lambda region: [{"title": "Big story", "source": "NDTV", "url": "u"}])
    calls = _fake_ddgs(monkeypatch, [])
    out = registry.call("web_search", {"query": "top news headlines", "mode": "news"}, ctx)
    assert out["results"][0]["title"] == "Big story" and out["region"] == "in-en" and not calls
    assert "don't" not in out["tip"].lower() or "card" in out["tip"]


def test_topic_news_is_local_retried_and_falls_back_worldwide(registry, ctx, monkeypatch):
    from ddgs import DDGS  # noqa: F401 - make sure the real package exists before faking it

    monkeypatch.setattr(web, "home_region", lambda: "in-en")
    row = {"title": "Cricket win", "source": "X", "url": "u", "date": "2026-11-17", "body": "India won the match."}
    calls = _fake_ddgs(monkeypatch, [RuntimeError("No results found."), [row]])
    out = registry.call("web_search", {"query": "cricket score", "mode": "news"}, ctx)
    assert out["results"][0]["title"] == "Cricket win" and calls[0][2]["region"] == "in-en"
    assert calls[0][2]["timelimit"] == "d" and len(calls) == 2  # retried once after an empty answer

    calls = _fake_ddgs(monkeypatch, [[], [], [], [], [row]])
    out = registry.call("web_search", {"query": "cricket score", "mode": "news"}, ctx)
    assert out["region"] == "wt-wt" and calls[-1][2]["region"] == "wt-wt"

    _fake_ddgs(monkeypatch, [])
    out = registry.call("web_search", {"query": "zzqx", "mode": "web"}, ctx)
    assert "No results" in out["result"]
