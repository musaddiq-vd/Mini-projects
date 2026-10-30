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


@tool(
    "web_search",
    "Search the web for current information. Use mode 'news' for recent news.",
    {
        "query": {"type": "string"},
        "mode": {"type": "string", "enum": ["web", "news"]},
    },
    required=["query"],
    slow=True,
)
def web_search(ctx, query: str, mode: str = "web"):
    from ddgs import DDGS

    ddgs = DDGS()
    if mode == "news":
        rows = ddgs.news(query, max_results=6) or []
        items = [{"title": r.get("title"), "source": r.get("source"),
                  "date": r.get("date"), "snippet": r.get("body")} for r in rows]
    else:
        rows = ddgs.text(query, max_results=6) or []
        items = [{"title": r.get("title"), "url": r.get("href"),
                  "snippet": r.get("body")} for r in rows]
    if not items:
        return "No results found."
    return {"results": items}


def _geocode(city: str) -> dict | None:
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}, timeout=TIMEOUT,
    )
    r.raise_for_status()
    results = r.json().get("results") or []
    return results[0] if results else None


@tool(
    "get_weather",
    "Current weather and today's forecast for a city. If no city is given, use the user's city from memory.",
    {"city": {"type": "string"}},
    required=["city"],
    slow=True,
)
def get_weather(ctx, city: str):
    place = _geocode(city)
    if not place:
        return {"error": f"Could not find a place called '{city}'."}
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
