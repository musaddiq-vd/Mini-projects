"""Smart home by voice: "bedroom light on", "fan off", "AC 24 pe", "lights blue", "sab lights band"."""

from __future__ import annotations

from jarvis.home import AC_MODES, HomeError, color_value, get_home
from jarvis.tools import tool

SETUP = ("No smart home is set up yet. In Settings > Home add Home Assistant (address + token) or your Tuya / "
         "Smart Life devices (Wipro, Syska, Havells ...).")
ACTIONS = ["on", "off", "toggle", "brightness", "color", "temperature", "mode", "status"]


@tool(
    "smart_home",
    "Control smart home devices: lights, plugs, fans, AC, curtains and scenes. device: its name or room "
    "('bedroom light', 'fan', 'AC') or 'all lights' / 'everything'. action: on, off, toggle, brightness (value "
    "0-100), color (value: red, warm white, blue ...), temperature (value: 16-30 for an AC), mode (value: cool, "
    "heat, fan, dry, auto), status.",
    {"device": {"type": "string"}, "action": {"type": "string", "enum": ACTIONS},
     "value": {"type": "string", "description": "brightness %, colour name, temperature or AC mode"}},
    required=["device", "action"],
)
def smart_home(ctx, device: str, action: str, value: str = ""):
    home = get_home(ctx.settings)
    if not home.configured():
        return {"error": SETUP}
    action = (action or "").lower()
    if action not in ACTIONS:
        return {"error": f"action must be one of: {', '.join(ACTIONS)}."}
    try:
        targets = home.find(device)
    except HomeError as exc:
        return {"error": str(exc)}
    if not targets:
        names = ", ".join(d.name for d in home.devices()[:12])
        return {"error": f"No device called '{device}'. I know: {names or 'none yet'}."}
    arg = None
    if action == "brightness":
        try:
            arg = max(1, min(100, int(float(str(value).rstrip("% ")))))
        except ValueError:
            return {"error": "Brightness needs a number from 1 to 100."}
    elif action == "color":
        arg = color_value(value)
        if arg is None:
            return {"error": f"I don't know the colour '{value}'. Try red, blue, warm white or a #hex code."}
    elif action == "temperature":
        try:
            arg = float(str(value).replace("°", "").replace("c", "").strip())
        except ValueError:
            return {"error": "Say the temperature as a number, like 24."}
        if not 16 <= arg <= 30:
            return {"error": "AC temperature should be between 16 and 30."}
    elif action == "mode":
        arg = str(value).lower().strip()
        if arg not in AC_MODES:
            return {"error": f"AC mode must be one of: {', '.join(AC_MODES)}."}
    if action == "status":
        out = {}
        for d in targets[:10]:
            try:
                out[d.name] = home.status(d)
            except HomeError as exc:
                out[d.name] = {"error": str(exc)}
        return {"status": out}
    done, problems = home.act(targets, action, arg)
    if ctx.events is not None and done:
        ctx.events.emit("smart_home", action=action, devices=done)
    if not done:
        return {"error": "; ".join(problems[:3])}
    if action == "brightness":
        what = f"at {arg}%"
    elif action == "color":
        what = f"set to {value}"
    elif action == "temperature":
        what = f"at {arg:g}°"
    elif action == "mode":
        what = f"on {arg} mode"
    else:
        what = {"on": "on", "off": "off", "toggle": "switched"}[action]
    result = {"result": f"{', '.join(done[:6])}{' and more' if len(done) > 6 else ''}: {what}."}
    if problems:
        result["problems"] = problems[:5]
    return result


@tool("smart_devices", "List the smart home devices Jarvis can control (name, type).")
def smart_devices(ctx):
    home = get_home(ctx.settings)
    if not home.configured():
        return {"error": SETUP}
    try:
        devices = home.devices(refresh=True)
    except HomeError as exc:
        return {"error": str(exc)}
    if not devices:
        return {"result": "Connected, but no devices were found."}
    return {"devices": [{"name": d.name, "type": d.kind, "via": "Home Assistant" if d.provider == "ha" else "Tuya"}
                        for d in devices[:60]], "count": len(devices)}
