"""Game launchers, flight search, Google Maps."""

from __future__ import annotations

import os
import platform
import urllib.parse
import webbrowser

from jarvis.tools import tool

OS = platform.system()

LAUNCHER_URIS = {
    ("steam", "updates"): "steam://open/downloads",
    ("steam", "library"): "steam://open/games",
    ("steam", "store"): "steam://store",
    ("epic", "updates"): "com.epicgames.launcher://store/library",
    ("epic", "library"): "com.epicgames.launcher://store/library",
    ("epic", "store"): "com.epicgames.launcher://store",
}


@tool(
    "game_launcher",
    "Open Steam or Epic Games at the downloads/updates page, the library or the store.",
    {
        "launcher": {"type": "string", "enum": ["steam", "epic"]},
        "page": {"type": "string", "enum": ["updates", "library", "store"]},
    },
    required=["launcher"],
)
def game_launcher(ctx, launcher: str, page: str = "updates"):
    uri = LAUNCHER_URIS.get((launcher.lower(), (page or "updates").lower()))
    if not uri:
        return {"error": "launcher must be steam or epic; page updates, library or store."}
    try:
        if OS == "Windows":
            os.startfile(uri)  # noqa: S606 - fixed launcher URIs
        else:
            webbrowser.open(uri)
    except OSError:
        return {"error": f"{launcher.capitalize()} doesn't seem to be installed."}
    return f"Opened {launcher.capitalize()} {page}."


@tool(
    "find_flights",
    "Search flights on Google Flights, e.g. from Mumbai to Dubai on a date. Opens the results in the browser.",
    {
        "origin": {"type": "string"},
        "destination": {"type": "string"},
        "date": {"type": "string", "description": "Departure date, e.g. 2026-11-20 or 'next friday'"},
        "return_date": {"type": "string"},
    },
    required=["origin", "destination"],
)
def find_flights(ctx, origin: str, destination: str, date: str = "", return_date: str = ""):
    q = f"flights from {origin} to {destination}"
    if date:
        q += f" on {date}"
    if return_date:
        q += f" returning {return_date}"
    webbrowser.open("https://www.google.com/travel/flights?q=" + urllib.parse.quote_plus(q))
    return f"Opened Google Flights for {origin} to {destination}. Use web_search with the same query for prices."


@tool(
    "open_map",
    "Show a place or directions on Google Maps.",
    {"place": {"type": "string"}, "from_place": {"type": "string", "description": "Start point for directions"}},
    required=["place"],
)
def open_map(ctx, place: str, from_place: str = ""):
    if from_place:
        url = "https://www.google.com/maps/dir/?api=1&" + urllib.parse.urlencode(
            {"origin": from_place, "destination": place})
    else:
        url = "https://www.google.com/maps/search/?api=1&" + urllib.parse.urlencode({"query": place})
    webbrowser.open(url)
    return f"Opened Google Maps for {place}."
