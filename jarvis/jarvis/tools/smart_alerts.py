"""Alerts the user asks for: "tell me when the battery is full", "batana jab battery 20% ho", "let me know when
the internet is back", "tell me when Chrome closes", "warn me when C drive has less than 10 GB".

Alerts are kept in state and checked by the background services every half minute. Each one rings once and
is then removed, unless the user asked for it to repeat.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from jarvis.tools import tool

KINDS = ["battery_below", "battery_full", "charger_unplugged", "charger_plugged", "internet_back", "internet_lost",
         "app_closed", "app_opened", "disk_below", "ram_below"]
MAX_ALERTS = 20
CHECK_SECONDS = 30


def _load(state) -> list[dict]:
    return list(state.get("smart_alerts", []) or []) if state is not None else []


def _save(state, items: list[dict]) -> None:
    state.set("smart_alerts", items)


def _describe(a: dict) -> str:
    v = a.get("value")
    return {
        "battery_below": f"battery at or below {v}%", "battery_full": "battery full (unplug the charger)",
        "charger_unplugged": "charger unplugged", "charger_plugged": "charger plugged in",
        "internet_back": "internet back", "internet_lost": "internet lost",
        "app_closed": f"{v} closed", "app_opened": f"{v} opened",
        "disk_below": f"{v} GB or less free on {a.get('drive') or 'C:'}", "ram_below": f"memory use below {v}%",
    }.get(a["kind"], a["kind"])


# -- what the PC is doing now (replaced in tests) ------------------------------------------------------
def _battery():
    import psutil

    b = psutil.sensors_battery()
    return (b.percent, bool(b.power_plugged)) if b else None


def _online() -> bool:
    from jarvis.tools.pc_health import PROBES, _tcp_ms

    return any(_tcp_ms(host, port, timeout=1.5) is not None for host, port, _ in PROBES[:2])


def _apps() -> set[str]:
    import psutil

    names = set()
    for p in psutil.process_iter(["name"]):
        n = (p.info.get("name") or "").lower().removesuffix(".exe")
        if n:
            names.add(n)
    return names


def _free_gb(drive: str) -> float:
    import psutil

    return psutil.disk_usage(drive or "C:\\").free / 1024 ** 3


def _ram() -> float:
    import psutil

    return psutil.virtual_memory().percent


PROBES: dict[str, Callable] = {"battery": _battery, "online": _online, "apps": _apps, "free_gb": _free_gb,
                               "ram": _ram}


def _app_running(name: str, apps: set[str]) -> bool:
    want = name.lower().removesuffix(".exe").replace(" ", "")
    return any(want in a.replace(" ", "") for a in apps)


def _met(a: dict, cache: dict) -> bool:
    kind, v = a["kind"], a.get("value")

    def get(what, *args):
        key = (what, *args)
        if key not in cache:
            cache[key] = PROBES[what](*args)
        return cache[key]

    if kind.startswith(("battery", "charger")):
        b = get("battery")
        if b is None:
            return False
        percent, plugged = b
        return {"battery_below": not plugged and percent <= float(v or 20),
                "battery_full": plugged and percent >= float(v or 100),
                "charger_unplugged": not plugged, "charger_plugged": plugged}[kind]
    if kind == "internet_back":
        return get("online")
    if kind == "internet_lost":
        return not get("online")
    if kind == "app_closed":
        return not _app_running(str(v), get("apps"))
    if kind == "app_opened":
        return _app_running(str(v), get("apps"))
    if kind == "disk_below":
        return get("free_gb", a.get("drive") or "C:\\") <= float(v)
    if kind == "ram_below":
        return get("ram") < float(v)
    return False


def check(state, now: float | None = None) -> list[str]:
    """Messages for alerts whose condition is met now. One-shot alerts are removed; repeating ones wait until
    the condition has gone away and comes back. Called by the background services."""
    items = _load(state)
    if not items:
        return []
    now = time.time() if now is None else now
    cache: dict = {}
    keep, messages, changed = [], [], False
    for a in items:
        try:
            met = _met(a, cache)
        except Exception:  # noqa: BLE001 - a sensor that fails once is tried again next time
            keep.append(a)
            continue
        if met and not a.get("armed_off"):
            messages.append(a.get("message") or f"Alert: {_describe(a)}.")
            changed = True
            if a.get("repeat"):
                keep.append({**a, "armed_off": True, "last": now})
            continue
        if not met and a.get("armed_off"):
            a = {k: x for k, x in a.items() if k != "armed_off"}
            changed = True
        keep.append(a)
    if changed:
        _save(state, keep)
    return messages


@tool(
    "add_alert",
    "Tell the user when something happens on the PC: battery_below (value = percent, default 20), battery_full, "
    "charger_unplugged / charger_plugged, internet_back / internet_lost, app_closed / app_opened (value = app "
    "name, e.g. 'chrome' for 'tell me when the download app closes'), disk_below (value = GB, drive like 'D:'), "
    "ram_below (value = percent). Rings once unless repeat=true. For alerts at a time use set_reminder.",
    {"kind": {"type": "string", "enum": KINDS}, "value": {"type": "string"}, "drive": {"type": "string"},
     "repeat": {"type": "boolean"}, "message": {"type": "string", "description": "what to say when it happens"}},
    required=["kind"],
)
def add_alert(ctx, kind: str, value: str = "", drive: str = "", repeat: bool = False, message: str = ""):
    if ctx.state is None:
        return {"error": "Alerts need the full Jarvis app."}
    if kind not in KINDS:
        return {"error": f"Unknown alert '{kind}'."}
    if kind in ("app_closed", "app_opened") and not value.strip():
        return {"error": "Which app should I watch?"}
    if kind in ("disk_below", "ram_below", "battery_below") and value:
        try:
            float(value)
        except ValueError:
            return {"error": f"'{value}' should be a number."}
    if kind == "disk_below" and not value:
        return {"error": "How many GB? For example 'warn me when C drive has less than 10 GB'."}
    items = _load(ctx.state)
    if len(items) >= MAX_ALERTS:
        return {"error": f"There are already {MAX_ALERTS} alerts. Cancel one first."}
    new_id = max((a["id"] for a in items), default=0) + 1
    alert = {"id": new_id, "kind": kind, "value": value.strip(), "repeat": bool(repeat), "created": time.time()}
    if drive:
        alert["drive"] = drive.strip().rstrip("\\") + "\\"
    if message:
        alert["message"] = message.strip()[:200]
    if kind == "app_closed" and not _app_running(value, PROBES["apps"]()):
        return {"error": f"'{value}' isn't running right now, so it can't close. Check the name."}
    _save(ctx.state, items + [alert])
    ctx.undo.push(f"alert {new_id}", lambda: _save(ctx.state, [a for a in _load(ctx.state) if a["id"] != new_id]))
    return {"result": f"Okay, I'll tell you when: {_describe(alert)}" + (" (every time)." if repeat else "."),
            "id": new_id}


@tool("list_alerts", "The alerts that are waiting (battery, internet, apps, disk).")
def list_alerts(ctx):
    items = _load(ctx.state)
    if not items:
        return "No alerts set."
    return {"alerts": [{"id": a["id"], "when": _describe(a), "repeat": a.get("repeat", False)} for a in items]}


@tool("cancel_alert", "Cancel an alert by its number (0 = all).", {"id": {"type": "integer"}}, required=["id"])
def cancel_alert(ctx, id: int):  # noqa: A002 - matches the tool argument name
    items = _load(ctx.state)
    keep = [] if int(id) == 0 else [a for a in items if a["id"] != int(id)]
    if len(keep) == len(items):
        return {"error": f"There's no alert number {id}."}
    _save(ctx.state, keep)
    return f"Cancelled {'all alerts' if int(id) == 0 else f'alert {id}'}."
