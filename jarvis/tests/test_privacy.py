"""Privacy: encrypted API key, masked logs, history retention, capture notices, the Privacy tab."""

import json
import logging
import sys
import time

import pytest

from jarvis import secure


# -- API key at rest --------------------------------------------------------------------------
@pytest.mark.skipif(not secure.available(), reason="DPAPI is Windows-only")
def test_api_key_is_encrypted_on_disk(isolated_home):
    from jarvis.config import Settings, load_settings, save_settings, settings_path

    save_settings(Settings(api_key="AIzaSECRET123"))
    raw = settings_path().read_text(encoding="utf-8")
    assert "AIzaSECRET123" not in raw and '"dpapi:' in raw
    assert load_settings().api_key == "AIzaSECRET123"


@pytest.mark.skipif(not secure.available(), reason="DPAPI is Windows-only")
def test_old_plain_key_is_encrypted_when_loaded(isolated_home):
    from jarvis.config import load_settings, settings_path

    settings_path().write_text(json.dumps({"api_key": "AIzaOLDPLAIN"}), encoding="utf-8")
    assert load_settings().api_key == "AIzaOLDPLAIN"
    assert "AIzaOLDPLAIN" not in settings_path().read_text(encoding="utf-8")


def test_a_key_from_another_account_is_dropped():
    assert secure.unprotect("dpapi:bm90IHJlYWxseQ==") == ""
    assert secure.unprotect("plain") == "plain" and secure.protect("") == ""


# -- logs ---------------------------------------------------------------------------------------
def test_logs_never_hold_secrets():
    from jarvis.logs import RedactFilter

    record = logging.LogRecord("jarvis", logging.WARNING, __file__, 1, "sending %s", ("OTP 123456",), None)
    RedactFilter().filter(record)
    assert "123456" not in record.getMessage() and "[OTP hidden]" in record.getMessage()
    try:
        raise ValueError("card 4111 1111 1111 1111 declined")
    except ValueError:
        record = logging.LogRecord("jarvis", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    RedactFilter().filter(record)
    assert "4111" not in record.exc_text


# -- history --------------------------------------------------------------------------------------
def test_old_history_is_deleted(tmp_path):
    from jarvis.history import ChatHistory

    path = tmp_path / "h.jsonl"
    old = time.time() - 40 * 86400
    path.write_text(json.dumps({"t": old, "role": "user", "text": "old"}) + "\n"
                    + json.dumps({"t": time.time(), "role": "user", "text": "new"}) + "\n", encoding="utf-8")
    h = ChatHistory(path)
    assert h.prune(30) == 1 and [i["text"] for i in h.recent()] == ["new"]
    assert h.prune(0) == 0


def test_history_can_be_turned_off(isolated_home):
    from jarvis.app import create_app
    from jarvis.config import Settings

    app = create_app(Settings(save_history=False))
    app.events.emit("user", text="don't keep this")
    assert app.history.recent() == []
    app.settings.save_history = True  # takes effect at once
    app.events.emit("user", text="keep this")
    assert [i["text"] for i in app.history.recent()] == ["keep this"]


# -- screen / camera -------------------------------------------------------------------------------
def test_screen_capture_is_announced(registry, ctx, monkeypatch):
    import jarvis.tools.vision as vision
    from jarvis.events import Events

    monkeypatch.setattr(vision, "screenshot_jpeg", lambda monitor=1: b"\xff\xd8JPEG")
    ctx.attach_image = lambda jpeg: None
    ctx.events = Events()
    seen = []
    ctx.events.subscribe(lambda k, d: seen.append(d.get("what")) if k == "capture" else None)
    registry.call("look_at_screen", {}, ctx)
    assert seen == ["screen"]


def test_camera_asks_first(registry, ctx, answers):
    ctx.attach_image = lambda jpeg: None
    answers.append(False)
    out = registry.call("look_at_camera", {}, ctx)
    assert "didn't allow" in out["result"]


def test_seen_cache_is_small_and_cleared(isolated_home, ctx):
    import jarvis.tools.vision as vision

    for _ in range(9):
        vision._keep_seen(ctx, b"img")
        time.sleep(0.002)
    folder = isolated_home / "cache"
    assert len(list(folder.glob("seen_*.jpg"))) <= vision.KEEP_SEEN
    assert vision.clear_seen_cache() >= 1 and not list(folder.glob("seen_*.jpg"))


# -- the Privacy tab and window --------------------------------------------------------------------
@pytest.fixture
def ui(monkeypatch, isolated_home):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from jarvis.app import create_app
    from jarvis.config import Settings

    qt = QApplication.instance() or QApplication(sys.argv)
    app = create_app(Settings())
    return app, qt


def test_privacy_tab_shows_and_deletes(ui):
    import copy

    from jarvis.ui.privacy import PrivacyTab

    app, qt = ui
    app.events.emit("user", text="hello")
    app.memory.remember("city", "Pune")
    app.state.set("notes", [{"id": 1, "text": "milk", "done": False}])
    app.english.record("d1-vocab", "vocab", 3, 5)
    s = copy.deepcopy(app.settings)
    tab = PrivacyTab(app, s)
    info = {k: lbl.text() for k, (lbl, _) in tab.rows.items()}
    assert info["history"].startswith("1 message ") and info["memory"] == "1 fact" and info["lists"] == "1 item"
    tab._clear_all()
    tab.refresh()
    assert app.history.recent() == [] and app.memory.all() == [] and app.state.get("notes") == []
    assert app._english is None and not (app.english.path.exists())
    tab.history_days.setValue(7)
    tab.ask_camera.setChecked(False)
    tab.apply_to(s)
    assert s.history_days == 7 and s.ask_camera is False


def test_settings_dialog_has_privacy(ui):
    from jarvis.ui.settings_dialog import SettingsDialog

    app, qt = ui
    d = SettingsDialog(app)
    assert any("Privacy" in d.tabs.tabText(i) for i in range(d.tabs.count()))


def test_capture_flash_and_mic_hint(ui):
    from jarvis.ui.window import Bridge, MainWindow

    app, qt = ui
    w = MainWindow(app, Bridge())
    w.show()
    app.events.emit("capture", what="screen")
    qt.processEvents()
    assert any("looked at your screen" in t.findChildren(type(w.title))[1].text() for t in w.toasts.toasts)
    app.events.emit("state", state="sleeping")
    assert "mic off" in w.hint.text() or "online" in w.hint.text() or "offline" in w.hint.text()
