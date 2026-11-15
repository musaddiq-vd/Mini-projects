"""Everyday helpers that work without the internet: a calculator, unit converter, world clock,
countdowns to a date, and coin / dice / random picks."""

from __future__ import annotations

import ast
import math
import operator
import random
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from jarvis.tools import tool

# -- calculator ---------------------------------------------------------------------------------------
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "sin": lambda x: math.sin(math.radians(x)), "cos": lambda x: math.cos(math.radians(x)),
          "tan": lambda x: math.tan(math.radians(x)), "log": math.log10, "ln": math.log, "abs": abs,
          "round": round, "floor": math.floor, "ceil": math.ceil, "factorial": math.factorial}
_CONSTS = {"pi": math.pi, "e": math.e}
_WORDS = [  # spoken maths -> symbols (English and common Hinglish)
    (r"\bplus\b|\baur\b|\bjod\b", "+"), (r"\bminus\b|\bghata\b", "-"),
    (r"\btimes\b|\bmultiplied by\b|\binto\b|\bguna\b|×|\bx\b", "*"),
    (r"\bdivided by\b|\bover\b|\bbhag\b|÷", "/"), (r"\bto the power of\b|\bpower\b|\^", "**"),
    (r"\bsquared\b", "**2"), (r"\bcubed\b", "**3"), (r"\bsquare root of\b|\broot\b", "sqrt"),
    (r"\blakh\b", "*100000"), (r"\bcrore\b", "*10000000"), (r"\bthousand\b|\bhazaar\b", "*1000"),
    (r"\bmillion\b", "*1000000"), (r"\bbillion\b", "*1000000000"),
]


def _prepare(expr: str) -> str:
    text = expr.lower().replace(",", "").strip().rstrip("?=. ")
    text = re.sub(r"^(what is|what's|calculate|kitna hai|kitna hota hai|batao)\s+", "", text)
    text = re.sub(r"\s+(?:kitna hai|kitna hota hai|kitna hoga|kya hai|hota hai|is what)$", "", text)
    text = re.sub(r"\s*(?:percent|pratishat)\b", "%", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:ka|ki|ke)\s*(\d+(?:\.\d+)?)\s*%", r"\1*(\2/100)", text)  # "2500 ka 18%"
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*(?:of|ka|ki|ke)\s*", r"(\1/100)*", text)  # "18% of 2500"
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", text)
    for pattern, sym in _WORDS:
        text = re.sub(pattern, sym, text)
    return text


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 1000 or abs(left) > 1e6 and right > 10):
            raise ValueError("That number is too big to work out.")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        args = [_eval(a) for a in node.args]
        if node.func.id == "factorial" and (args[0] > 500 or args[0] != int(args[0])):
            raise ValueError("Factorial works for whole numbers up to 500.")
        return _FUNCS[node.func.id](*[int(a) if node.func.id == "factorial" else a for a in args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError("I can only work out plain maths.")


def _nice(x: float) -> str:
    if isinstance(x, int) or float(x).is_integer():
        return f"{int(x):,}"
    return f"{x:,.6f}".rstrip("0").rstrip(".")


def calc(expression: str) -> float:
    tree = ast.parse(_prepare(expression), mode="eval")
    return _eval(tree)


@tool(
    "calculate",
    "Work out maths exactly instead of guessing: sums, percentages ('18% of 2500'), powers, square roots, "
    "trig in degrees, log, factorial, lakh/crore. Use it for any arithmetic, tips, splits and discounts.",
    {"expression": {"type": "string", "description": "e.g. '2500 * 18%', '(1200+450)/3', 'sqrt(144)'"}},
    required=["expression"],
)
def calculate(ctx, expression: str):
    try:
        value = calc(expression)
    except ZeroDivisionError:
        return {"error": "Can't divide by zero."}
    except (ValueError, SyntaxError, TypeError, OverflowError) as exc:
        msg = str(exc) if isinstance(exc, ValueError) else "I couldn't understand that sum."
        return {"error": msg}
    return {"expression": expression, "result": _nice(value), "value": value}


# -- units --------------------------------------------------------------------------------------------
# every unit -> (dimension, factor to the base unit)
_UNITS: dict[str, tuple[str, float]] = {}


def _add(dim: str, factor: float, *names: str) -> None:
    for n in names:
        _UNITS[n] = (dim, factor)


_add("length", 1, "m", "meter", "meters", "metre", "metres")
_add("length", 1000, "km", "kilometer", "kilometers", "kilometre", "kilometres")
_add("length", 0.01, "cm", "centimeter", "centimeters")
_add("length", 0.001, "mm", "millimeter", "millimeters")
_add("length", 1609.344, "mi", "mile", "miles")
_add("length", 0.3048, "ft", "foot", "feet")
_add("length", 0.0254, "in", "inch", "inches")
_add("length", 0.9144, "yd", "yard", "yards")
_add("mass", 1, "kg", "kilo", "kilos", "kilogram", "kilograms")
_add("mass", 0.001, "g", "gram", "grams")
_add("mass", 0.45359237, "lb", "lbs", "pound", "pounds")
_add("mass", 0.028349523125, "oz", "ounce", "ounces")
_add("mass", 1000, "t", "ton", "tonne", "tonnes")
_add("volume", 1, "l", "liter", "liters", "litre", "litres")
_add("volume", 0.001, "ml", "milliliter", "milliliters", "millilitre")
_add("volume", 3.785411784, "gal", "gallon", "gallons")
_add("volume", 0.2365882365, "cup", "cups")
_add("speed", 1, "kmh", "km/h", "kph", "kmph")
_add("speed", 1.609344, "mph")
_add("speed", 3.6, "m/s", "mps")
_add("data", 1, "b", "byte", "bytes")
_add("data", 1024, "kb", "kilobyte", "kilobytes")
_add("data", 1024 ** 2, "mb", "megabyte", "megabytes")
_add("data", 1024 ** 3, "gb", "gigabyte", "gigabytes")
_add("data", 1024 ** 4, "tb", "terabyte", "terabytes")
_add("time", 1, "s", "sec", "second", "seconds")
_add("time", 60, "min", "minute", "minutes")
_add("time", 3600, "h", "hr", "hour", "hours", "ghanta", "ghante")
_add("time", 86400, "day", "days", "din")
_add("time", 604800, "week", "weeks", "hafta", "hafte")
_add("area", 1, "sqm", "m2", "square meter", "square meters")
_add("area", 0.09290304, "sqft", "ft2", "square foot", "square feet")
_add("area", 4046.8564224, "acre", "acres")
_add("area", 10000, "hectare", "hectares", "ha")
_TEMPS = {"c": "c", "celsius": "c", "centigrade": "c", "f": "f", "fahrenheit": "f", "k": "k", "kelvin": "k"}


def _unit(name: str) -> str:
    return re.sub(r"\s+", " ", name.lower().strip().replace("°", "").replace("degrees ", "").replace("degree ", ""))


def convert(value: float, from_unit: str, to_unit: str) -> float:
    a, b = _unit(from_unit), _unit(to_unit)
    if a in _TEMPS and b in _TEMPS:
        c = {"c": value, "f": (value - 32) * 5 / 9, "k": value - 273.15}[_TEMPS[a]]
        return {"c": c, "f": c * 9 / 5 + 32, "k": c + 273.15}[_TEMPS[b]]
    if a not in _UNITS or b not in _UNITS:
        missing = a if a not in _UNITS else b
        raise ValueError(f"I don't know the unit '{missing}'.")
    (da, fa), (db, fb) = _UNITS[a], _UNITS[b]
    if da != db:
        raise ValueError(f"Can't turn {da} into {db}.")
    return value * fa / fb


@tool(
    "convert_units",
    "Convert between units: length (km, miles, feet, inches), weight (kg, pounds, grams), volume (litres, "
    "gallons, cups), temperature (C, F, K), speed (km/h, mph), data (MB, GB, TB), time and area (sqft, acres). "
    "For money use convert_currency.",
    {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
    required=["value", "from_unit", "to_unit"],
)
def convert_units(ctx, value: float, from_unit: str, to_unit: str):
    try:
        out = convert(float(value), from_unit, to_unit)
    except ValueError as exc:
        return {"error": str(exc)}
    return {"result": f"{_nice(value)} {from_unit} = {_nice(round(out, 4))} {to_unit}", "value": out}


# -- world clock --------------------------------------------------------------------------------------
_CITY_ZONES = {
    "india": "Asia/Kolkata", "delhi": "Asia/Kolkata", "mumbai": "Asia/Kolkata", "pune": "Asia/Kolkata",
    "bangalore": "Asia/Kolkata", "bengaluru": "Asia/Kolkata", "chennai": "Asia/Kolkata", "hyderabad": "Asia/Kolkata",
    "pakistan": "Asia/Karachi", "lahore": "Asia/Karachi", "islamabad": "Asia/Karachi",
    "dubai": "Asia/Dubai", "uae": "Asia/Dubai", "abu dhabi": "Asia/Dubai", "saudi": "Asia/Riyadh",
    "riyadh": "Asia/Riyadh", "jeddah": "Asia/Riyadh", "mecca": "Asia/Riyadh", "makkah": "Asia/Riyadh",
    "qatar": "Asia/Qatar", "doha": "Asia/Qatar", "kuwait": "Asia/Kuwait", "oman": "Asia/Muscat",
    "uk": "Europe/London", "england": "Europe/London", "france": "Europe/Paris", "germany": "Europe/Berlin",
    "usa": "America/New_York", "us": "America/New_York", "new york": "America/New_York", "nyc": "America/New_York",
    "washington": "America/New_York", "boston": "America/New_York", "san francisco": "America/Los_Angeles",
    "california": "America/Los_Angeles", "seattle": "America/Los_Angeles", "texas": "America/Chicago",
    "dallas": "America/Chicago", "houston": "America/Chicago", "canada": "America/Toronto",
    "australia": "Australia/Sydney", "japan": "Asia/Tokyo", "china": "Asia/Shanghai", "beijing": "Asia/Shanghai",
    "singapore": "Asia/Singapore", "bangladesh": "Asia/Dhaka", "nepal": "Asia/Kathmandu", "sri lanka": "Asia/Colombo",
    "russia": "Europe/Moscow", "turkey": "Europe/Istanbul", "egypt": "Africa/Cairo", "kenya": "Africa/Nairobi",
}


def find_zone(place: str) -> str | None:
    key = place.lower().strip()
    if key in _CITY_ZONES:
        return _CITY_ZONES[key]
    want = key.replace(" ", "_")
    for zone in sorted(available_timezones()):
        if zone.lower().split("/")[-1] == want:
            return zone
    return None


@tool(
    "world_time",
    "The current time in another city or country, and how far ahead or behind the user it is.",
    {"place": {"type": "string", "description": "a city or country, e.g. 'London', 'Dubai', 'New York'"}},
    required=["place"],
)
def world_time(ctx, place: str):
    zone = find_zone(place)
    if not zone:
        return {"error": f"I don't know the time zone of '{place}'. Try a big city nearby."}
    try:
        there = datetime.now(ZoneInfo(zone))
    except ZoneInfoNotFoundError:
        return {"error": "Time zone data is missing on this computer."}
    here = datetime.now().astimezone()
    diff_h = (there.utcoffset() - here.utcoffset()).total_seconds() / 3600
    if diff_h == 0:
        rel = "the same time as you"
    else:
        h = abs(diff_h)
        amount = f"{int(h)} hour{'s' if int(h) != 1 else ''}" + (" 30 minutes" if h % 1 == 0.5 else
                                                                  " 45 minutes" if h % 1 == 0.75 else "")
        rel = f"{amount} {'ahead of' if diff_h > 0 else 'behind'} you"
    return {"place": place, "zone": zone, "time": there.strftime("%I:%M %p").lstrip("0"),
            "day": there.strftime("%A %d %B"), "compared": rel}


# -- countdown ----------------------------------------------------------------------------------------
@tool(
    "days_until",
    "How many days until (or since) a date: birthdays, exams, trips, festivals. Accepts '25 december', "
    "'2026-12-31', 'next friday', 'in 3 weeks'.",
    {"when": {"type": "string"}, "what": {"type": "string", "description": "what the date is, e.g. 'my exam'"}},
    required=["when"],
)
def days_until(ctx, when: str, what: str = ""):
    from jarvis.dates import parse_day

    today = date.today()
    day = parse_day(when, today=today, past=False)
    if day is None:
        return {"error": f"I couldn't understand the date '{when}'."}
    n = (day - today).days
    label = what or day.strftime("%d %B %Y")
    if n == 0:
        text = f"{label} is today!"
    elif n > 0:
        weeks = f" (about {n // 7} weeks)" if n >= 14 else ""
        text = f"{n} day{'s' if n != 1 else ''} until {label}{weeks}."
    else:
        text = f"{label} was {-n} day{'s' if n != -1 else ''} ago."
    return {"date": day.isoformat(), "weekday": day.strftime("%A"), "days": n, "result": text}


# -- random -------------------------------------------------------------------------------------------
_rng = random.SystemRandom()


@tool(
    "random_pick",
    "Flip a coin, roll dice, pick a random number, or choose one option from a list ('pizza or biryani?').",
    {"kind": {"type": "string", "enum": ["coin", "dice", "number", "choose"]},
     "options": {"type": "array", "items": {"type": "string"}},
     "count": {"type": "integer", "description": "how many dice"},
     "low": {"type": "integer"}, "high": {"type": "integer"}},
    required=["kind"],
)
def random_pick(ctx, kind: str, options: list | None = None, count: int = 1, low: int = 1, high: int = 100):
    kind = (kind or "").lower()
    if kind == "coin":
        return {"result": _rng.choice(["Heads", "Tails"])}
    if kind == "dice":
        count = max(1, min(int(count or 1), 10))
        rolls = [_rng.randint(1, 6) for _ in range(count)]
        return {"result": ", ".join(map(str, rolls)), "rolls": rolls, "total": sum(rolls)}
    if kind == "number":
        low, high = sorted((int(low), int(high)))
        return {"result": _rng.randint(low, high), "range": [low, high]}
    if kind == "choose":
        opts = [str(o).strip() for o in (options or []) if str(o).strip()]
        if len(opts) < 2:
            return {"error": "Give me at least two options to choose from."}
        return {"result": _rng.choice(opts), "options": opts}
    return {"error": f"Unknown kind '{kind}'."}


# -- stopwatch ----------------------------------------------------------------------------------------
_watch: dict = {}  # used when there is no state store (text mode tests)


def _clock() -> float:
    import time

    return time.time()


def _fmt_secs(secs: float) -> str:
    secs = int(round(secs))
    h, rest = divmod(secs, 3600)
    m, s = divmod(rest, 60)
    parts = [f"{h} hour{'s' if h != 1 else ''}"] if h else []
    if m:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    if s or not parts:
        parts.append(f"{s} second{'s' if s != 1 else ''}")
    return " ".join(parts)


@tool(
    "stopwatch",
    "A stopwatch that counts up: start, lap, check how long it has been, or stop ('stopwatch start karo', "
    "'kitna time hua?'). For counting down use set_timer.",
    {"action": {"type": "string", "enum": ["start", "lap", "status", "stop"]}},
    required=["action"],
)
def stopwatch(ctx, action: str):
    st = ctx.state
    data = (st.get("stopwatch") if st is not None else _watch.get("sw")) or {}
    t = _clock()

    def save(value):
        if st is not None:
            st.set("stopwatch", value)
        else:
            _watch["sw"] = value

    action = (action or "").lower()
    if action == "start":
        save({"start": t, "laps": []})
        return {"result": "Stopwatch started."}
    if not data.get("start"):
        return {"error": "The stopwatch isn't running. Say 'start the stopwatch' first."}
    elapsed = t - data["start"]
    if action == "lap":
        laps = list(data.get("laps", [])) + [elapsed]
        save({**data, "laps": laps})
        return {"result": f"Lap {len(laps)}: {_fmt_secs(elapsed)}.", "lap": len(laps), "seconds": round(elapsed, 1)}
    if action == "stop":
        save({})
        laps = data.get("laps") or []
        return {"result": f"Stopped at {_fmt_secs(elapsed)}.", "seconds": round(elapsed, 1),
                "laps": [round(x, 1) for x in laps]}
    return {"result": f"{_fmt_secs(elapsed)} so far.", "seconds": round(elapsed, 1)}
