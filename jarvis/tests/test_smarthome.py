"""Smart home: Home Assistant (real HTTP against a local fake) and Tuya (a fake device library)."""

import http.server
import json
import sys
import threading

import pytest

from jarvis.home import Device, Home, HomeAssistant, HomeError, TuyaLocal, color_value

STATES = [
    {"entity_id": "light.bedroom", "state": "off", "attributes": {"friendly_name": "Bedroom Light"}},
    {"entity_id": "light.hall", "state": "on", "attributes": {"friendly_name": "Hall Lamp", "brightness": 200}},
    {"entity_id": "climate.living_ac", "state": "cool", "attributes": {"friendly_name": "Living Room AC",
                                                                        "temperature": 24}},
    {"entity_id": "fan.study", "state": "off", "attributes": {"friendly_name": "Study Fan"}},
    {"entity_id": "scene.movie", "state": "scening", "attributes": {"friendly_name": "Movie Time"}},
    {"entity_id": "sensor.temp", "state": "21", "attributes": {}},
]


@pytest.fixture
def fake_ha():
    calls = []

    class H(http.server.BaseHTTPRequestHandler):
        def _ok(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def _auth(self):
            if self.headers.get("Authorization") != "Bearer good-token":
                self.send_response(401)
                self.end_headers()
                return False
            return True

        def do_GET(self):  # noqa: N802
            if not self._auth():
                return
            if self.path == "/api/":
                return self._ok({"message": "API running."})
            if self.path == "/api/states":
                return self._ok(STATES)
            if self.path.startswith("/api/states/"):
                eid = self.path.rsplit("/", 1)[-1]
                return self._ok(next(s for s in STATES if s["entity_id"] == eid))
            self.send_response(404)
            self.end_headers()

        def do_POST(self):  # noqa: N802
            if not self._auth():
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body))
            self._ok([])

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", calls
    srv.shutdown()


class FakeTuyaDevice:
    log = []
    fail = None

    def __init__(self, dev_id, ip, key):
        self.dev_id = dev_id

    def set_version(self, v):
        pass

    def set_socketTimeout(self, s):  # noqa: N802
        pass

    def set_socketRetryLimit(self, n):  # noqa: N802
        pass

    def _do(self, what):
        FakeTuyaDevice.log.append((self.dev_id, what))
        return FakeTuyaDevice.fail or {"dps": {"1": True}}

    def turn_on(self):
        return self._do("on")

    def turn_off(self):
        return self._do("off")

    def status(self):
        return FakeTuyaDevice.fail or {"dps": {"1": True}}

    def set_brightness_percentage(self, v):
        return self._do(f"brightness {v}")

    def set_colour(self, r, g, b):
        return self._do(f"colour {r},{g},{b}")

    def set_white_percentage(self, v):
        return self._do(f"white {v}")

    def set_value(self, dps, v):
        return self._do(f"dps{dps}={v}")


class FakeTinytuya:
    BulbDevice = OutletDevice = Device = FakeTuyaDevice


TUYA = [{"name": "Desk Lamp", "type": "light", "id": "bf01", "key": "k1", "ip": "192.168.1.20", "version": 3.3},
        {"name": "Heater Plug", "type": "plug", "id": "bf02", "key": "k2", "room": "bedroom"},
        {"name": "Split AC", "type": "ac", "id": "bf03", "key": "k3", "temp_dps": 2},
        {"name": "No key", "type": "plug", "id": "bf04"}]


@pytest.fixture
def tuya():
    FakeTuyaDevice.log.clear()
    FakeTuyaDevice.fail = None
    return TuyaLocal(TUYA, lib=FakeTinytuya)


def test_home_assistant(fake_ha):
    url, calls = fake_ha
    ha = HomeAssistant(url, "good-token")
    assert ha.check() == "API running."
    devices = {d.name: d for d in ha.devices()}
    assert set(devices) == {"Bedroom Light", "Hall Lamp", "Living Room AC", "Study Fan", "Movie Time"}
    assert devices["Living Room AC"].kind == "ac" and devices["Movie Time"].kind == "scene"
    ha.call(devices["Bedroom Light"], "on")
    ha.call(devices["Bedroom Light"], "brightness", 40)
    ha.call(devices["Bedroom Light"], "color", (255, 0, 0))
    ha.call(devices["Living Room AC"], "temperature", 23)
    ha.call(devices["Living Room AC"], "mode", "fan")
    ha.call(devices["Movie Time"], "off")  # a scene can only be started
    assert calls == [
        ("/api/services/light/turn_on", {"entity_id": "light.bedroom"}),
        ("/api/services/light/turn_on", {"entity_id": "light.bedroom", "brightness_pct": 40}),
        ("/api/services/light/turn_on", {"entity_id": "light.bedroom", "rgb_color": [255, 0, 0]}),
        ("/api/services/climate/set_temperature", {"entity_id": "climate.living_ac", "temperature": 23.0}),
        ("/api/services/climate/set_hvac_mode", {"entity_id": "climate.living_ac", "hvac_mode": "fan_only"}),
        ("/api/services/scene/turn_on", {"entity_id": "scene.movie"}),
    ]
    assert ha.status(devices["Hall Lamp"]) == {"state": "on", "brightness": 200}


def test_home_assistant_errors(fake_ha):
    url, _ = fake_ha
    with pytest.raises(HomeError, match="refused the token"):
        HomeAssistant(url, "wrong").devices()
    with pytest.raises(HomeError, match="Couldn't reach"):
        HomeAssistant("http://127.0.0.1:1", "x").devices()


def test_tuya(tuya):
    devices = {d.name: d for d in tuya.devices()}
    assert set(devices) == {"Desk Lamp", "Heater Plug", "Split AC"}  # rows without a key are skipped
    tuya.call(devices["Desk Lamp"], "brightness", 30)
    tuya.call(devices["Desk Lamp"], "color", (255, 255, 255))
    tuya.call(devices["Desk Lamp"], "color", (0, 90, 255))
    tuya.call(devices["Split AC"], "temperature", 24)
    tuya.call(devices["Heater Plug"], "toggle")
    assert FakeTuyaDevice.log == [("bf01", "brightness 30"), ("bf01", "white 100"), ("bf01", "colour 0,90,255"),
                                  ("bf03", "dps2=24"), ("bf02", "off")]
    with pytest.raises(HomeError, match="only be switched on and off"):
        tuya.call(devices["Heater Plug"], "brightness", 50)
    FakeTuyaDevice.fail = {"Error": "Network Error: Unable to Connect"}
    with pytest.raises(HomeError, match="didn't answer"):
        tuya.call(devices["Desk Lamp"], "on")


def test_find_devices(tuya):
    home = Home({"tuya": tuya})
    names = lambda q: sorted(d.name for d in home.find(q))  # noqa: E731
    assert names("desk lamp") == ["Desk Lamp"] and names("the lamp") == ["Desk Lamp"]
    assert names("bedroom") == ["Heater Plug"]  # by room
    assert names("all lights") == ["Desk Lamp"] and names("sab lights") == ["Desk Lamp"]
    assert names("everything") == ["Desk Lamp", "Heater Plug", "Split AC"]
    assert names("AC") == ["Split AC"] and names("split ac") == ["Split AC"]
    assert names("desk lmp") == ["Desk Lamp"]  # a small typo
    assert names("garage door") == [] and names("") == []


def test_colors():
    assert color_value("Warm White") == (255, 190, 120) and color_value("neela") == (0, 90, 255)
    assert color_value("#ff8800") == (255, 136, 0) and color_value("sparkly") is None


@pytest.fixture
def hctx(ctx, tuya, monkeypatch):
    from jarvis import home as home_mod
    from jarvis.events import Events

    monkeypatch.setattr(home_mod, "get_home", lambda s: Home({"tuya": tuya}))
    import jarvis.tools.smarthome as sh

    monkeypatch.setattr(sh, "get_home", lambda s: Home({"tuya": tuya}))
    ctx.events = Events()
    return ctx


def test_tool(registry, hctx):
    out = registry.call("smart_home", {"device": "desk lamp", "action": "on"}, hctx)
    assert out["result"] == "Desk Lamp: on."
    out = registry.call("smart_home", {"device": "all lights", "action": "color", "value": "warm white"}, hctx)
    assert "set to warm white" in out["result"]
    assert "between 16 and 30" in registry.call("smart_home", {"device": "ac", "action": "temperature",
                                                               "value": "40"}, hctx)["error"]
    assert "1 to 100" in registry.call("smart_home", {"device": "lamp", "action": "brightness",
                                                      "value": "lots"}, hctx)["error"]
    assert "I know" in registry.call("smart_home", {"device": "garage", "action": "on"}, hctx)["error"]
    assert "colour" in registry.call("smart_home", {"device": "lamp", "action": "color", "value": "sparkly"},
                                     hctx)["error"]
    status = registry.call("smart_home", {"device": "lamp", "action": "status"}, hctx)["status"]
    assert status == {"Desk Lamp": {"state": "on"}}
    out = registry.call("smart_home", {"device": "everything", "action": "brightness", "value": "50"}, hctx)
    assert "Desk Lamp" in out["result"] and len(out["problems"]) == 2  # plugs and the AC can't dim
    assert registry.call("smart_devices", {}, hctx)["count"] == 3


def test_not_set_up(registry, ctx):
    assert "Settings > Home" in registry.call("smart_home", {"device": "lamp", "action": "on"}, ctx)["error"]


def test_secrets_encrypted_and_not_backed_up(isolated_home):
    from jarvis import backup, secure
    from jarvis.config import Settings, load_settings, save_settings, settings_path

    save_settings(Settings(ha_url="http://ha.local:8123", ha_token="tok-123", tuya_devices=[TUYA[0]]))
    raw = settings_path().read_text(encoding="utf-8")
    if secure.available():
        assert "tok-123" not in raw and '"k1"' not in raw
    s = load_settings()
    assert s.ha_token == "tok-123" and s.tuya_devices[0]["key"] == "k1"
    assert {"ha_token", "tuya_devices"} <= set(backup.SECRET_SETTINGS)


def test_offline_switching():
    from jarvis.offline import parse

    assert parse("turn on the bedroom light").args == {"device": "bedroom light", "action": "on"}
    assert parse("AC band karo").args == {"device": "ac", "action": "off"}
    assert parse("turn off wifi") is None and parse("volume band karo").tool == "volume"


def test_settings_tab_import(monkeypatch, isolated_home, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from jarvis.config import Settings
    from jarvis.ui.home_tab import HomeTab

    qt = QApplication.instance() or QApplication(sys.argv)
    tab = HomeTab(Settings(tuya_devices=[TUYA[1]]))
    f = tmp_path / "devices.json"
    f.write_text(json.dumps([{"name": "Bedroom Bulb", "id": "aa1", "key": "x1", "ip": "10.0.0.5", "version": "3.4"},
                             {"name": "Heater Plug", "id": "bf02", "key": "k2"},  # already there
                             {"name": "broken"}]), encoding="utf-8")
    assert tab.import_file(str(f)) == 1
    tab.ha_url.setText("homeassistant.local")  # no http://: not saved
    s = Settings()
    tab.apply_to(s)
    assert s.ha_url == "" and [d["name"] for d in s.tuya_devices] == ["Heater Plug", "Bedroom Bulb"]
    assert s.tuya_devices[1]["type"] == "light" and s.tuya_devices[1]["version"] == 3.4
    (tmp_path / "bad.json").write_text("nope", encoding="utf-8")
    assert tab.import_file(str(tmp_path / "bad.json")) == 0 and qt is not None
