"""Smart energy manager: battery health and drain, power-hungry apps, brightness and the Windows power plan,
with suggestions and rough savings. "How is my battery health?", "battery jaldi khatam ho rahi hai",
"switch to power saver".

Nothing here changes a setting without the user's "yes": power_mode always asks first (and can be undone).
The battery report comes from ``powercfg /batteryreport /xml``, written into a private temp folder that is
deleted right after; only a few numbers are read from it with plain patterns (no XML entities are expanded).
"""

from __future__ import annotations

import platform
import re
import subprocess
import tempfile
import time
from pathlib import Path

from jarvis.tools import tool

OS = platform.system()
NO_WINDOW = 0x08000000 if OS == "Windows" else 0
# the plans that ship with Windows
PLANS = {
    "balanced": "381b4222-f694-41f0-9685-ff5bb260df2e",
    "power_saver": "a1841308-3541-4fab-bc81-f71556f20b4a",
    "high_performance": "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",
}
PLAN_NAMES = {"balanced": "Balanced", "power_saver": "Power saver", "high_performance": "High performance"}
MAX_REPORT_BYTES = 5_000_000
SKIP_PROCESSES = {"system idle process", "idle", "system", "registry", "memory compression", "secure system"}


# -- reading the PC (replaced in tests) -------------------------------------------------------------
def run_powercfg(args: list[str], timeout: int = 25) -> subprocess.CompletedProcess:
    return subprocess.run(["powercfg", *args], capture_output=True, text=True, timeout=timeout,
                          creationflags=NO_WINDOW)


def read_battery():
    import psutil

    b = psutil.sensors_battery()
    if not b:
        return None
    left = b.secsleft if isinstance(b.secsleft, (int, float)) and b.secsleft > 0 else None
    return {"percent": round(float(b.percent)), "plugged": bool(b.power_plugged),
            "minutes_left": round(left / 60) if left and left < 10 ** 6 else None}


def read_brightness() -> int | None:
    from jarvis.tools.settings_tools import get_brightness

    return get_brightness()


def read_processes() -> list[dict]:
    """[{name, cpu_seconds, age_hours}] for every process that can be read."""
    import psutil

    out, now = [], time.time()
    for p in psutil.process_iter(["name", "cpu_times", "create_time"]):
        try:
            t = p.info.get("cpu_times")
            if t is None:
                continue
            out.append({"name": p.info.get("name") or "", "cpu_seconds": float(t.user + t.system),
                        "age_hours": max((now - float(p.info.get("create_time") or now)) / 3600, 0.05)})
        except (psutil.Error, OSError, TypeError, ValueError):
            continue
    return out


# -- parsing ----------------------------------------------------------------------------------------
def _number(text: str) -> int | None:
    digits = re.sub(r"[^\d]", "", text or "")
    return int(digits) if digits else None


def parse_battery_report(text: str) -> dict:
    """Design and full-charge capacity (mWh) and cycle count from the XML (or HTML) battery report."""
    design = [int(x) for x in re.findall(r"<DesignCapacity>\s*(\d+)\s*</DesignCapacity>", text or "")]
    full = [int(x) for x in re.findall(r"<FullChargeCapacity>\s*(\d+)\s*</FullChargeCapacity>", text or "")]
    cycles = [int(x) for x in re.findall(r"<CycleCount>\s*(\d+)\s*</CycleCount>", text or "")]
    if not design:  # the HTML report: "DESIGN CAPACITY</span></td><td>45,000 mWh"
        flat = re.sub(r"<[^>]+>", " ", text or "")
        design = [n for n in (_number(m) for m in re.findall(r"DESIGN CAPACITY\s+([\d,.]+)\s*mWh", flat, re.I)) if n]
        full = [n for n in (_number(m) for m in re.findall(r"FULL CHARGE CAPACITY\s+([\d,.]+)\s*mWh", flat, re.I))
                if n]
        cycles = [n for n in (_number(m) for m in re.findall(r"CYCLE COUNT\s+([\d,.]+)", flat, re.I))
                  if n is not None]
    out: dict = {}
    if design and full:
        d, f = sum(design), sum(full)
        out.update(design_mwh=d, full_charge_mwh=f, health_percent=round(min(f / d, 1.5) * 100) if d else None)
    if cycles and any(cycles):
        out["cycles"] = sum(cycles)
    return out


def parse_active_scheme(text: str) -> tuple[str, str] | None:
    """'Power Scheme GUID: 381b...  (Balanced)' -> (guid, 'Balanced')."""
    m = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\s*\(([^)]*)\)", text or "", re.I)
    return (m.group(1).lower(), m.group(2).strip()) if m else None


def battery_report() -> dict:
    """Battery health from powercfg, via a private temp folder that is removed afterwards. {} if unavailable."""
    if OS != "Windows":
        return {}
    with tempfile.TemporaryDirectory(prefix="jarvis-battery-") as folder:
        out = Path(folder) / "battery.xml"
        try:
            done = run_powercfg(["/batteryreport", "/xml", "/output", str(out)])
        except (OSError, subprocess.SubprocessError):
            return {}
        if done.returncode != 0 or not out.is_file() or out.stat().st_size > MAX_REPORT_BYTES:
            return {}
        raw = out.read_bytes()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return parse_battery_report(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    return {}


def active_plan() -> dict | None:
    if OS != "Windows":
        return None
    try:
        done = run_powercfg(["/getactivescheme"], timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    found = parse_active_scheme(done.stdout)
    if not found:
        return None
    guid, name = found
    key = next((k for k, g in PLANS.items() if g == guid), "custom")
    return {"guid": guid, "name": name, "plan": key}


def power_hungry(procs: list[dict], count: int = 5) -> list[dict]:
    """Apps using the most CPU time per hour they've been running (grouped by name)."""
    grouped: dict[str, dict] = {}
    for p in procs:
        name = str(p.get("name", "")).removesuffix(".exe") or "unknown"
        if name.lower() in SKIP_PROCESSES:
            continue
        row = grouped.setdefault(name, {"name": name, "cpu_seconds": 0.0, "rate": 0.0})
        row["cpu_seconds"] += p["cpu_seconds"]
        row["rate"] += p["cpu_seconds"] / 60 / p["age_hours"]  # CPU minutes per hour
    total = sum(r["cpu_seconds"] for r in grouped.values()) or 1.0
    rows = sorted(grouped.values(), key=lambda r: r["rate"], reverse=True)[:count]
    return [{"name": r["name"], "cpu_minutes_per_hour": round(r["rate"], 1),
             "cpu_share_percent": round(r["cpu_seconds"] / total * 100, 1)} for r in rows]


def drain_rate(state) -> float | None:
    """Percent per hour lost on battery, from the readings the proactive engine keeps."""
    if state is None:
        return None
    from datetime import datetime

    from jarvis import proactive

    d = proactive._data(state)
    log = d["battery_log"]
    if not log:
        return None
    last = log[-1]
    snap = proactive.Snapshot(now=datetime.fromtimestamp(last[0]), battery=(last[1], bool(last[2])),
                              battery_log=log)
    rate = proactive.battery_drain(snap)
    return round(rate, 1) if rate else None


def suggestions(report: dict, threshold: int = 25) -> list[dict]:
    """Ways to save power, each with a rough saving. Estimates assume a typical laptop where the screen uses
    about a quarter of the power and the CPU most of the rest."""
    tips = []
    bat = report.get("battery") or {}
    on_battery = bool(bat) and not bat.get("plugged")
    bright = report.get("brightness")
    plan = (report.get("power_plan") or {}).get("plan")
    if on_battery and isinstance(bright, int) and bright > 50:
        saving = round((bright - 40) / 100 * 25)
        tips.append({"tip": f"Lower the brightness from {bright}% to 40%.",
                     "saving": f"about {saving}% longer battery", "action": "brightness 40"})
    if on_battery and plan == "high_performance":
        tips.append({"tip": "Switch from High performance to Balanced on battery.",
                     "saving": "about 10-20% longer battery", "action": "power_mode balanced"})
    if on_battery and plan == "balanced" and bat.get("percent", 100) <= threshold + 10:
        tips.append({"tip": "Switch to Power saver until you can charge.",
                     "saving": "about 10-15% longer battery", "action": "power_mode power_saver"})
    for app in (report.get("top_apps") or [])[:3]:
        if app["cpu_minutes_per_hour"] >= 15:
            tips.append({"tip": f"{app['name']} keeps the CPU busy ({app['cpu_minutes_per_hour']} CPU-min per "
                         "hour). Close it if you don't need it.", "saving": "about 5-15% longer battery",
                         "action": f"close_app {app['name']}"})
    health = (report.get("health") or {}).get("health_percent")
    if isinstance(health, int) and health < 80:
        tips.append({"tip": f"The battery now holds {health}% of its design capacity. Avoid keeping it at 100% "
                     "all day (use the maker's smart-charging option) and consider a service check.",
                     "saving": "slows further wear"})
    rate = report.get("discharge_percent_per_hour")
    if on_battery and rate and rate >= 20:
        tips.append({"tip": f"The battery is draining fast (~{rate}%/hour). Close unused apps and tabs, and turn "
                     "off Bluetooth / keyboard backlight if you don't need them.", "saving": "varies"})
    return tips


@tool(
    "energy_report",
    "Battery and power: battery health (design vs full-charge capacity, charge cycles), how fast it drains, the "
    "most power-hungry apps, screen brightness and the Windows power plan, with suggestions and rough savings. "
    "Use for 'battery health', 'battery jaldi khatam ho rahi hai', 'how can I save battery'.",
    slow=True,
)
def energy_report(ctx):
    report: dict = {}
    try:
        report["battery"] = read_battery()
    except Exception:  # noqa: BLE001 - every part is optional
        report["battery"] = None
    try:
        report["health"] = battery_report()
    except Exception:  # noqa: BLE001
        report["health"] = {}
    rate = drain_rate(getattr(ctx, "state", None))
    if rate and report["battery"] and not report["battery"]["plugged"]:
        report["discharge_percent_per_hour"] = rate
    try:
        report["brightness"] = read_brightness()
    except Exception:  # noqa: BLE001
        report["brightness"] = None
    report["power_plan"] = active_plan()
    try:
        report["top_apps"] = power_hungry(read_processes())
    except Exception:  # noqa: BLE001
        report["top_apps"] = []
    report["suggestions"] = suggestions(report, int(getattr(ctx.settings, "battery_saver_percent", 25) or 25))
    if report["battery"] is None:
        report["note"] = "No battery found (a desktop PC?) - power plan and apps only."
    report["tip"] = ("Summarise in two or three sentences and offer the top suggestion. Never change a setting "
                     "unless the user says yes; power_mode asks them on screen.")
    return report


@tool(
    "power_mode",
    "Switch the Windows power plan: balanced, power_saver or high_performance (asks the user first; can be "
    "undone). mode 'status' just says which plan is active.",
    {"mode": {"type": "string", "enum": ["status", *PLANS]}},
    required=["mode"],
)
def power_mode(ctx, mode: str):
    mode = str(mode).strip().lower().replace(" ", "_")
    if OS != "Windows":
        return {"error": "Power plans can only be changed on Windows."}
    current = active_plan()
    if mode == "status":
        return {"plan": current["name"] if current else "unknown"}
    if mode not in PLANS:
        return {"error": f"Unknown mode '{mode}'. Use balanced, power_saver or high_performance."}
    if current and current["guid"] == PLANS[mode]:
        return f"The {PLAN_NAMES[mode]} plan is already on."
    before = current["name"] if current else "the current plan"
    if not ctx.confirm.confirm(f"Switch the Windows power plan from {before} to {PLAN_NAMES[mode]}?"):
        return "Cancelled - the user did not confirm."
    try:
        done = run_powercfg(["/setactive", PLANS[mode]], timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"Couldn't change the power plan: {exc}"}
    if done.returncode != 0:
        return {"error": f"The {PLAN_NAMES[mode]} plan isn't available on this PC (many Windows 11 laptops only "
                         "have Balanced; use Settings > System > Power > Power mode, or Battery saver)."}
    if current:
        old = current["guid"]
        ctx.undo.push(f"power plan {PLAN_NAMES[mode]}", lambda: run_powercfg(["/setactive", old], timeout=10))
    return f"Switched to the {PLAN_NAMES[mode]} plan. Say 'undo' to go back to {before}."


TIPS = [
    "Lower the screen brightness - the screen is one of the biggest power users.",
    "Turn on Battery saver (Settings > System > Power & battery) or switch to the Power saver plan.",
    "Close apps and browser tabs you aren't using, especially video, games and heavy websites.",
    "Turn off Bluetooth, the keyboard backlight and extra monitors when you don't need them.",
    "Use dark mode with a dark wallpaper on OLED screens.",
    "Unplug USB devices you aren't using.",
    "For battery life over years: avoid sitting at 100% all day; many laptops have a charge limit option.",
]


@tool("battery_saver_tips", "Quick tips to make the battery last longer, plus what applies right now.")
def battery_saver_tips(ctx):
    out: dict = {"tips": TIPS}
    try:
        bat = read_battery()
    except Exception:  # noqa: BLE001
        bat = None
    if bat:
        out["battery"] = bat
        threshold = int(getattr(ctx.settings, "battery_saver_percent", 25) or 25)
        if not bat["plugged"] and bat["percent"] <= threshold:
            out["offer"] = ("The battery is low: offer to switch to power saver (power_mode) and lower the "
                            "brightness to 40 - only if the user says yes.")
    out["tip"] = "Give the two or three most useful tips briefly."
    return out

