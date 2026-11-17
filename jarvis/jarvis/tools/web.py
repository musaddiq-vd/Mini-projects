"""Web search (DuckDuckGo) and weather (Open-Meteo) - no API keys needed."""

from __future__ import annotations

import requests

from jarvis.tools import tool

TIMEOUT = 10

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
    75: "heavy snow", 80: "rain showers", 81: "rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


# Windows time zone name -> DuckDuckGo region, so "top news" means news from where the user lives
_TZ_REGIONS = {"india": "in-en", "pakistan": "pk-en", "bangladesh": "bd-en", "nepal": "in-en", "sri lanka": "in-en",
               "arabian": "xa-en", "arab": "xa-en", "gmt standard": "uk-en", "w. europe": "de-de",
               "romance": "fr-fr", "aus ": "au-en", "singapore": "sg-en", "tokyo": "jp-jp", "china": "cn-zh",
               "eastern standard": "us-en", "central standard": "us-en", "pacific standard": "us-en",
               "mountain standard": "us-en", "canada": "ca-en", "south africa": "za-en", "e. africa": "ke-en"}


def home_region() -> str:
    """The DuckDuckGo region for the user's own country, from the PC's time zone ("wt-wt" = worldwide)."""
    import time

    zone = (time.tzname[0] or "").lower()
    return next((region for key, region in _TZ_REGIONS.items() if key in zone), "wt-wt")


def readable(text: str | None) -> str:
    """The text, or "" when it is scrambled (some sites hide their snippets as encoded junk)."""
    text = (text or "").strip()
    if not text:
        return ""
    words = text.split()
    real = sum(1 for w in words if w.strip(".,;:!?'\"()-").isalpha())
    return text if words and real / len(words) >= 0.6 else ""


def _search(kind: str, query: str, region: str, attempts: int = 2, timelimit: str | None = None) -> list[dict]:
    from ddgs import DDGS

    last: Exception | None = None
    for _ in range(attempts):  # the search service sometimes answers empty once and fine the next time
        try:
            ddgs = DDGS()
            extra = {"timelimit": timelimit} if timelimit else {}
            rows = (ddgs.news if kind == "news" else ddgs.text)(query, region=region, max_results=8, **extra) or []
            if rows:
                return rows
        except Exception as exc:  # noqa: BLE001 - "No results found" is an exception in ddgs
            last = exc
    if last and "no results" not in str(last).lower():
        raise last
    return []


# Top headlines come straight from the RSS feeds of well-known newsrooms in the user's country: search engines
# answer "top news" with old or foreign stories.
HEADLINE_FEEDS = {
    "in-en": ["https://www.thehindu.com/news/national/feeder/default.rss",
              "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
              "https://feeds.feedburner.com/ndtvnews-top-stories",
              "https://indianexpress.com/section/india/feed/"],
    "uk-en": ["https://feeds.bbci.co.uk/news/rss.xml"],
    "us-en": ["https://feeds.npr.org/1001/rss.xml"],
    "wt-wt": ["https://feeds.bbci.co.uk/news/world/rss.xml"],
}
_GENERIC_NEWS = {"top", "news", "headline", "headlines", "today", "today's", "todays", "latest", "breaking", "the",
                 "of", "day", "main", "big", "aaj", "ki", "ke", "ka", "khabar", "khabrein", "samachar", "taaza",
                 "current", "events", "world", "national", "india", "indian", "in"}
MAX_FEED_BYTES = 3_000_000
# topic newsrooms for when a search only finds old stories: region -> [(keywords, feed)]
TOPIC_FEEDS = {
    "in-en": [({"sensex", "nifty", "stock", "stocks", "share", "shares", "market", "markets", "ipo", "bse", "nse"},
               "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
              ({"business", "economy", "rupee", "gst", "rbi", "budget", "inflation"},
               "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms"),
              ({"cricket", "ipl", "sports", "sport", "football", "match", "test", "odi", "t20", "hockey"},
               "https://timesofindia.indiatimes.com/rssfeeds/4719148.cms")],
}
FEED_NAMES = {"thehindu.com": "The Hindu", "timesofindia": "Times of India", "economictimes": "Economic Times", "ndtv": "NDTV",
              "indianexpress.com": "Indian Express", "bbci.co.uk": "BBC", "npr.org": "NPR"}


def is_headlines_query(query: str) -> bool:
    """'top news', 'today's headlines', 'aaj ki khabar' - no particular topic."""
    import re

    words = re.findall(r"[a-z']+", (query or "").lower())
    return bool(words) and all(w in _GENERIC_NEWS for w in words)


def _feed_items(url: str) -> list[dict]:
    import html
    import re
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Jarvis/2.0 (personal news reader)"}, stream=True)
    r.raise_for_status()
    raw = r.raw.read(MAX_FEED_BYTES + 1, decode_content=True)
    if len(raw) > MAX_FEED_BYTES or b"<!ENTITY" in raw[:4000]:
        return []  # too big, or entity tricks: skip this feed
    root = ET.fromstring(raw)
    channel_title = next((name for key, name in FEED_NAMES.items() if key in url), "")
    if not channel_title and root.find("channel") is not None:
        channel_title = (root.find("channel").findtext("title") or "").split("|")[0].split(" - ")[0].strip()
    out = []
    for item in root.iter("item"):
        title = html.unescape((item.findtext("title") or "").strip())
        if not title:
            continue
        try:
            when = parsedate_to_datetime(item.findtext("pubDate") or "")
        except (TypeError, ValueError):
            when = None
        desc = html.unescape(re.sub(r"<[^>]+>", " ", item.findtext("description") or ""))
        out.append({"title": title, "source": channel_title, "url": (item.findtext("link") or "").strip(),
                    "date": when.isoformat() if when else "", "snippet": readable(" ".join(desc.split())[:300]),
                    "_ts": when.timestamp() if when else 0.0})
    return out


def top_headlines(region: str, limit: int = 6) -> list[dict]:
    """Newest headlines from the region's newsrooms, a few from each, without repeats."""
    feeds = HEADLINE_FEEDS.get(region) or HEADLINE_FEEDS["wt-wt"]
    per_feed: list[list[dict]] = []
    for url in feeds:
        try:
            per_feed.append(sorted(_feed_items(url), key=lambda x: -x["_ts"])[:limit])
        except Exception:  # noqa: BLE001 - one newsroom down never spoils the rest
            continue
    merged, seen = [], set()
    for row in range(limit):  # take turns between newsrooms, newest first in each
        for items in per_feed:
            if row < len(items):
                key = items[row]["title"].lower()[:60]
                if key not in seen:
                    seen.add(key)
                    merged.append(items[row])
    merged.sort(key=lambda x: -x["_ts"])
    picked, per_source = [], {}
    for x in merged:  # at most two from one newsroom, so the card shows several of them
        if per_source.get(x["source"], 0) < 2:
            per_source[x["source"]] = per_source.get(x["source"], 0) + 1
            picked.append(x)
    for x in picked:
        x.pop("_ts", None)
    return picked[:limit]


def _topic_feed(query: str, region: str, limit: int = 6) -> list[dict]:
    """Newest stories from a newsroom feed that matches the query's topic (sensex -> markets), or []."""
    import re

    words = set(re.findall(r"[a-z0-9]+", (query or "").lower()))
    for keywords, url in TOPIC_FEEDS.get(region, []):
        if words & keywords:
            try:
                items = sorted(_feed_items(url), key=lambda x: -x["_ts"])[:limit]
            except Exception:  # noqa: BLE001
                return []
            for x in items:
                x.pop("_ts", None)
            return items
    return []


def _fresh(rows: list[dict], days: int = 3) -> bool:
    """True when at least one row is from the last few days."""
    from datetime import datetime, timedelta, timezone

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()[:10]
    return any(str(r.get("date") or "")[:10] >= cutoff for r in rows)


def _recent(rows: list[dict], days: int = 7, keep: int = 3) -> list[dict]:
    """News rows newest first; older than `days` only when there aren't `keep` recent ones."""
    from datetime import datetime, timedelta, timezone

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()[:10]
    rows = sorted(rows, key=lambda r: str(r.get("date") or ""), reverse=True)
    fresh = [r for r in rows if str(r.get("date") or "")[:10] >= cutoff]
    return fresh if len(fresh) >= keep else rows


@tool(
    "web_search",
    "Search the web for current information. Use mode 'news' for recent news. News and results come from the "
    "user's own country unless they ask about another place (then set region, e.g. 'us-en', 'uk-en', 'wt-wt').",
    {
        "query": {"type": "string"},
        "mode": {"type": "string", "enum": ["web", "news"]},
        "region": {"type": "string", "description": "optional, e.g. in-en, us-en, uk-en, wt-wt (worldwide)"},
    },
    required=["query"],
    slow=True,
)
def web_search(ctx, query: str, mode: str = "web", region: str = ""):
    region = (region or home_region()).lower()
    kind = "news" if mode == "news" else "text"
    if kind == "news" and is_headlines_query(query):
        items = top_headlines(region)
        if items:
            return {"results": items, "mode": "news", "query": "Top headlines", "region": region,
                    "tip": "These headlines are shown to the user as a card. Summarise the three most important "
                           "in a few spoken sentences."}
    rows = []
    if kind == "news":
        rows = _recent(_search(kind, query, region))
        if rows and not _fresh(rows) and "latest" not in query.lower():  # only old stories: ask for the latest
            newer = _recent(_search(kind, f"{query} latest", region))
            rows = newer if _fresh(newer) else rows
        if not _fresh(rows):  # still old: the matching newsroom feed has today's stories
            topical = _topic_feed(query, region)
            if topical:
                rows = [{"title": x["title"], "source": x["source"], "url": x["url"], "date": x["date"],
                         "body": x["snippet"]} for x in topical]
    else:
        rows = _search(kind, query, region)
    if not rows and region != "wt-wt":
        rows = _search(kind, query, "wt-wt")  # nothing local: try worldwide
        rows = _recent(rows) if kind == "news" else rows
        region = "wt-wt"
    if mode == "news":
        items = [{"title": r.get("title"), "source": r.get("source"), "url": r.get("url"),
                  "date": r.get("date"), "snippet": readable(r.get("body"))} for r in rows]
        items.sort(key=lambda x: str(x.get("date") or ""), reverse=True)  # newest first
    else:
        items = [{"title": r.get("title"), "url": r.get("href"), "snippet": readable(r.get("body"))} for r in rows]
    items = [x for x in items if x.get("title")][:6]
    if not items:
        return "No results found right now. Offer to try again or to search a more specific topic."
    out = {"results": items, "mode": mode, "query": query, "region": region,
           "tip": "These results are shown to the user as a card. Summarise the most important ones briefly; "
                  "don't say there are no results."}
    if mode == "news" and not _fresh(items):
        newest = str(items[0].get("date") or "")[:10]
        out["note"] = (f"These are older stories (the newest is from {newest or 'an unknown date'}). Say the date "
                       "so the user knows it isn't today's news.")
    return out


# old names and common spellings -> the name the weather service knows
CITY_ALIASES = {"bangalore": "Bengaluru", "banglore": "Bengaluru", "bengalore": "Bengaluru", "bombay": "Mumbai",
                "bambai": "Mumbai", "calcutta": "Kolkata", "madras": "Chennai", "gurgaon": "Gurugram",
                "poona": "Pune", "baroda": "Vadodara", "trivandrum": "Thiruvananthapuram", "cochin": "Kochi",
                "benaras": "Varanasi", "banaras": "Varanasi", "dilli": "Delhi", "new dehli": "New Delhi",
                "dehli": "Delhi", "hydrabad": "Hyderabad", "ahmadabad": "Ahmedabad", "allahabad": "Prayagraj",
                "mysore": "Mysuru", "mangalore": "Mangaluru", "simla": "Shimla", "pondicherry": "Puducherry"}
KNOWN_CITIES = sorted(set(CITY_ALIASES.values()) | {
    "Agra", "Ahmedabad", "Amritsar", "Aurangabad", "Bhopal", "Bhubaneswar", "Chandigarh", "Coimbatore",
    "Dehradun", "Goa", "Guwahati", "Indore", "Jaipur", "Jammu", "Jodhpur", "Kanpur", "Kota", "Lucknow", "Ludhiana",
    "Madurai", "Meerut", "Nagpur", "Nashik", "Noida", "Patna", "Raipur", "Rajkot", "Ranchi", "Srinagar", "Surat",
    "Thane", "Udaipur", "Vijayawada", "Visakhapatnam", "Karachi", "Lahore", "Islamabad", "Dhaka", "Kathmandu",
    "Colombo", "Dubai", "Abu Dhabi", "Riyadh", "Jeddah", "Doha", "Muscat", "London", "Paris", "Berlin",
    "New York", "San Francisco", "Los Angeles", "Chicago", "Toronto", "Sydney", "Melbourne", "Singapore",
    "Tokyo", "Bangkok", "Kuala Lumpur", "Hong Kong", "Istanbul", "Cairo", "Nairobi", "Johannesburg"})


def _lookup(name: str) -> dict | None:
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": name, "count": 1},
                     timeout=TIMEOUT)
    r.raise_for_status()
    results = r.json().get("results") or []
    return results[0] if results else None


def _geocode(city: str) -> dict | None:
    """A place for the weather service, forgiving old names and small spelling mistakes."""
    import difflib

    city = " ".join((city or "").split())
    if not city:
        return None
    place = _lookup(CITY_ALIASES.get(city.lower(), city))
    if place:
        return place
    close = difflib.get_close_matches(city.title(), KNOWN_CITIES, n=1, cutoff=0.75)
    return _lookup(close[0]) if close else None


def home_city(ctx) -> str:
    """The user's city: Settings first, then memory."""
    if getattr(ctx.settings, "city", ""):
        return ctx.settings.city
    for key in ("city", "home_city"):
        fact = ctx.memory.get(key) if ctx.memory is not None else None
        if fact:
            return fact.value
    return ""


@tool(
    "get_weather",
    "Current weather and today's forecast for a city. If no city is given, use the user's city from memory.",
    {"city": {"type": "string", "description": "leave empty for the user's own city"}},
    slow=True,
)
def get_weather(ctx, city: str = ""):
    city = (city or "").strip() or home_city(ctx)
    if not city:
        return {"error": "Which city? Ask the user, and remember it with `remember` as 'city' for next time."}
    place = _geocode(city)
    if not place:
        return {"error": f"Could not find a place called '{city}'. Ask the user to spell it or name a nearby city."}
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto", "forecast_days": 1,
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    cur, day = data["current"], data["daily"]
    return {
        "place": f"{place['name']}, {place.get('country', '')}".strip(", "),
        "now": WEATHER_CODES.get(cur["weather_code"], "unknown"),
        "temperature_c": cur["temperature_2m"],
        "feels_like_c": cur["apparent_temperature"],
        "humidity_percent": cur["relative_humidity_2m"],
        "wind_kmh": cur["wind_speed_10m"],
        "today_max_c": day["temperature_2m_max"][0],
        "today_min_c": day["temperature_2m_min"][0],
        "rain_chance_percent": day["precipitation_probability_max"][0],
    }
