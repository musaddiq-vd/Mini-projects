"""Smart home: lights, plugs, fans and ACs through Home Assistant or Tuya / Smart Life devices.

- Home Assistant: the URL and a long-lived access token; every light, switch, fan, climate (AC), cover and
  scene it knows becomes a device.
- Tuya / Smart Life (Wipro, Syska, Havells, Halonix ...): controlled directly on the home WiFi with tinytuya,
  so it is fast and keeps working without internet. Each device needs its id, IP and local key (Settings > Home).

Names are matched loosely ("bedroom light", "the fan"), and "all lights" means every light.
"""

from __future__ import annotations

import difflib
import logging
import re
import threading
import time
from dataclasses import dataclass, field

log = logging.getLogger("jarvis.home")

KINDS = ("light", "plug", "fan", "ac", "switch", "cover", "scene")
COLORS = {
    "red": (255, 0, 0), "green": (0, 200, 60), "blue": (0, 90, 255), "white": (255, 255, 255),
    "warm white": (255, 190, 120), "yellow": (255, 210, 0), "orange": (255, 120, 0), "purple": (150, 60, 255),
    "pink": (255, 80, 170), "cyan": (0, 220, 230),
    # common Hindi colour words
    "laal": (255, 0, 0), "lal": (255, 0, 0), "hara": (0, 200, 60), "neela": (0, 90, 255), "nila": (0, 90, 255),
    "safed": (255, 255, 255), "peela": (255, 210, 0), "narangi": (255, 120, 0), "gulabi": (255, 80, 170),
    "baingani": (150, 60, 255),
}
AC_MODES = ("cool", "heat", "fan", "dry", "auto")
TIMEOUT = 6


class HomeError(Exception):
    """A friendly reason a command didn't work."""


@dataclass
class Device:
    name: str
    kind: str
    provider: str  # "ha" | "tuya"
    ref: str  # entity_id or Tuya device id
    room: str = ""
    extra: dict = field(default_factory=dict)  # Tuya: ip, key, version, temp_dps ...

    def label(self) -> str:
        return f"{self.name} ({self.kind})"


# -- Home Assistant ------------------------------------------------------------------------------
HA_KINDS = {"light": "light", "switch": "plug", "fan": "fan", "climate": "ac", "cover": "cover", "scene": "scene",
            "input_boolean": "switch"}


class HomeAssistant:
    def __init__(self, url: str, token: str, http=None):
        self.url = url.rstrip("/")
        self.token = token
        if http is None:
            import requests

            http = requests
        self.http = http

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def _req(self, method: str, path: str, body: dict | None = None):
        try:
            resp = getattr(self.http, method)(self.url + path, headers=self._headers(), json=body, timeout=TIMEOUT)
        except Exception as exc:
            raise HomeError("Couldn't reach Home Assistant - is it on, and is the address right?") from exc
        code = getattr(resp, "status_code", 200)
        if code in (401, 403):
            raise HomeError("Home Assistant refused the token. Make a new long-lived token in your HA profile.")
        if code >= 400:
            raise HomeError(f"Home Assistant said no (error {code}).")
        return resp.json() if hasattr(resp, "json") else None

    def devices(self) -> list[Device]:
        out = []
        for s in self._req("get", "/api/states") or []:
            eid = s.get("entity_id", "")
            domain = eid.split(".", 1)[0]
            if domain not in HA_KINDS:
                continue
            name = (s.get("attributes") or {}).get("friendly_name") or eid.split(".", 1)[-1].replace("_", " ")
            out.append(Device(str(name), HA_KINDS[domain], "ha", eid))
        return out

    def check(self) -> str:
        data = self._req("get", "/api/")
        return (data or {}).get("message", "API running.")

    def call(self, d: Device, action: str, value=None) -> str:
        domain = d.ref.split(".", 1)[0]
        body: dict = {"entity_id": d.ref}
        if action in ("on", "off", "toggle"):
            if domain == "scene":
                service = "turn_on"
            elif domain == "cover":
                service = {"on": "open_cover", "off": "close_cover", "toggle": "toggle"}[action]
            else:
                service = {"on": "turn_on", "off": "turn_off", "toggle": "toggle"}[action]
        elif action == "brightness":
            domain, service = "light", "turn_on"
            body["brightness_pct"] = int(value)
        elif action == "color":
            domain, service = "light", "turn_on"
            body["rgb_color"] = list(value)
        elif action == "temperature":
            domain, service = "climate", "set_temperature"
            body["temperature"] = float(value)
        elif action == "mode":
            domain, service = "climate", "set_hvac_mode"
            body["hvac_mode"] = {"fan": "fan_only"}.get(value, value)
        else:
            raise HomeError(f"Can't do '{action}' on {d.name}.")
        self._req("post", f"/api/services/{domain}/{service}", body)
        return "ok"

    def status(self, d: Device) -> dict:
        s = self._req("get", f"/api/states/{d.ref}") or {}
        attrs = s.get("attributes") or {}
        out = {"state": s.get("state", "unknown")}
        for k in ("brightness", "current_temperature", "temperature", "hvac_mode"):
            if k in attrs:
                out[k] = attrs[k]
        return out


# -- Tuya / Smart Life on the local network ------------------------------------------------------
class TuyaLocal:
    def __init__(self, devices: list[dict], lib=None):
        self.config = [d for d in devices if d.get("id") and d.get("key")]
        self._lib = lib

    def lib(self):
        if self._lib is None:
            try:
                import tinytuya
            except ImportError as exc:
                raise HomeError("Tuya support isn't installed (pip install tinytuya).") from exc
            self._lib = tinytuya
        return self._lib

    def devices(self) -> list[Device]:
        out = []
        for c in self.config:
            kind = c.get("type", "plug") if c.get("type") in KINDS else "plug"
            out.append(Device(c.get("name") or c["id"], kind, "tuya", c["id"], c.get("room", ""),
                              {"ip": c.get("ip", "Auto"), "key": c["key"], "version": float(c.get("version") or 3.3),
                               "temp_dps": c.get("temp_dps"), "mode_dps": c.get("mode_dps")}))
        return out

    def _open(self, d: Device):
        lib = self.lib()
        cls = lib.BulbDevice if d.kind == "light" else lib.OutletDevice if d.kind == "plug" else lib.Device
        dev = cls(d.ref, d.extra.get("ip") or "Auto", d.extra["key"])
        dev.set_version(d.extra.get("version", 3.3))
        dev.set_socketTimeout(TIMEOUT)
        dev.set_socketRetryLimit(1)
        return dev

    @staticmethod
    def _check(result) -> None:
        if isinstance(result, dict) and result.get("Error"):
            msg = str(result.get("Error"))
            if "Network Error" in msg or "Unable to Connect" in msg or "Timeout" in msg:
                raise HomeError("The device didn't answer - is it on the same WiFi and switched on at the wall?")
            if "decrypt" in msg.lower() or "key" in msg.lower():
                raise HomeError("The device's local key looks wrong - copy it again into Settings > Home.")
            raise HomeError(f"The device said: {msg}")

    def call(self, d: Device, action: str, value=None) -> str:
        dev = self._open(d)
        if action in ("on", "off"):
            self._check(dev.turn_on() if action == "on" else dev.turn_off())
        elif action == "toggle":
            st = dev.status()
            self._check(st)
            on = bool((st.get("dps") or {}).get("1") or (st.get("dps") or {}).get("20"))
            self._check(dev.turn_off() if on else dev.turn_on())
        elif action == "brightness" and d.kind == "light":
            self._check(dev.set_brightness_percentage(int(value)))
        elif action == "color" and d.kind == "light":
            r, g, b = value
            if (r, g, b) == (255, 255, 255):
                self._check(dev.set_white_percentage(100))
            else:
                self._check(dev.set_colour(r, g, b))
        elif action == "temperature" and d.extra.get("temp_dps"):
            self._check(dev.set_value(int(d.extra["temp_dps"]), int(value)))
        elif action == "mode" and d.extra.get("mode_dps"):
            self._check(dev.set_value(int(d.extra["mode_dps"]), str(value)))
        else:
            raise HomeError(f"{d.name} can only be switched on and off here"
                            + (" (add its temperature setting number in Settings > Home)" if d.kind == "ac" else "")
                            + ".")
        return "ok"

    def status(self, d: Device) -> dict:
        st = self._open(d).status()
        self._check(st)
        dps = st.get("dps") or {}
        on = dps.get("1", dps.get("20"))
        return {"state": "on" if on else "off" if on is not None else "unknown"}


# -- the home ------------------------------------------------------------------------------------
class Home:
    """All configured devices, name lookup and commands."""

    CACHE_S = 300

    def __init__(self, providers: dict):
        self.providers = providers  # {"ha": HomeAssistant, "tuya": TuyaLocal}
        self._devices: list[Device] | None = None
        self._at = 0.0
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings) -> Home:
        providers = {}
        if getattr(settings, "ha_url", "") and getattr(settings, "ha_token", ""):
            providers["ha"] = HomeAssistant(settings.ha_url, settings.ha_token)
        tuya = [d for d in getattr(settings, "tuya_devices", []) or [] if isinstance(d, dict)]
        if tuya:
            providers["tuya"] = TuyaLocal(tuya)
        return cls(providers)

    def configured(self) -> bool:
        return bool(self.providers)

    def devices(self, refresh: bool = False) -> list[Device]:
        with self._lock:
            if self._devices is not None and not refresh and time.time() - self._at < self.CACHE_S:
                return self._devices
            found, errors = [], []
            for name, p in self.providers.items():
                try:
                    found += p.devices()
                except HomeError as exc:
                    errors.append(str(exc))
            if not found and errors:
                raise HomeError(errors[0])
            self._devices, self._at = found, time.time()
            return found

    def find(self, query: str) -> list[Device]:
        """'bedroom light' -> that device; 'all lights' -> every light; 'everything' -> all."""
        q = " ".join(re.sub(r"[^\w\s]", " ", (query or "").lower()).split())
        q = re.sub(r"^(?:the|my|meri|mera)\s+", "", q)
        devices = self.devices()
        if not q:
            return []
        if q in ("all", "everything", "sab", "sabhi", "all devices", "sab kuch"):
            return [d for d in devices if d.kind != "scene"]
        m = re.match(r"(?:all|sab|saari|sare|saare)\s+(?:the\s+)?(\w+?)s?$", q)
        if m:
            kind = {"light": "light", "lamp": "light", "bulb": "light", "batti": "light", "plug": "plug",
                    "socket": "plug", "fan": "fan", "pankha": "fan", "ac": "ac", "switch": "switch"}.get(m.group(1))
            if kind:
                return [d for d in devices if d.kind == kind]
        exact = [d for d in devices if d.name.lower() == q]
        if exact:
            return exact
        words = set(q.split())
        scored = []
        for d in devices:
            name_words = set(d.name.lower().split()) | ({d.room.lower()} if d.room else set())
            kind_words = {d.kind, {"light": "lamp", "ac": "aircon", "plug": "socket"}.get(d.kind, d.kind)}
            hits = len(words & name_words) * 2 + len(words & kind_words)
            if hits:
                scored.append((hits, d))
        if scored:
            best = max(h for h, _ in scored)
            return [d for h, d in scored if h == best]
        close = difflib.get_close_matches(q, [d.name.lower() for d in devices], n=1, cutoff=0.6)
        return [d for d in devices if close and d.name.lower() == close[0]]

    def act(self, targets: list[Device], action: str, value=None) -> tuple[list[str], list[str]]:
        """Run on each device; (done names, problems)."""
        done, problems = [], []
        for d in targets:
            try:
                self.providers[d.provider].call(d, action, value)
                done.append(d.name)
            except HomeError as exc:
                problems.append(f"{d.name}: {exc}")
            except Exception as exc:  # a device library error never crashes Jarvis
                log.exception("smart home %s failed", d.name)
                problems.append(f"{d.name}: {type(exc).__name__}")
        return done, problems

    def status(self, d: Device) -> dict:
        return self.providers[d.provider].status(d)


def color_value(name: str) -> tuple[int, int, int] | None:
    n = (name or "").strip().lower()
    if n in COLORS:
        return COLORS[n]
    m = re.fullmatch(r"#?([0-9a-f]{6})", n)
    if m:
        h = m.group(1)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return None


_home: dict = {"key": None, "home": None}


def get_home(settings) -> Home:
    """One Home per configuration (rebuilt when the settings change)."""
    key = (getattr(settings, "ha_url", ""), getattr(settings, "ha_token", ""),
           repr(getattr(settings, "tuya_devices", [])))
    if _home["key"] != key:
        _home["key"], _home["home"] = key, Home.from_settings(settings)
    return _home["home"]
