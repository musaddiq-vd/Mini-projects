"""Custom colours and colours from the wallpaper."""

import colorsys
import sys

import pytest
from PIL import Image


def hue(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hsv(r, g, b)


def test_two_colour_wallpaper():
    from jarvis.ui.wallcolors import palette

    img = Image.new("RGB", (200, 100), (230, 40, 40))  # mostly red ...
    img.paste((40, 60, 220), (140, 0, 200, 100))  # ... with some blue
    accent, second = palette(img)
    assert hue(accent)[0] < 0.05 or hue(accent)[0] > 0.95  # red leads (it covers more)
    assert 0.55 < hue(second)[0] < 0.75  # blue second
    assert hue(accent)[2] >= 0.84 and hue(accent)[1] >= 0.54  # bright enough to glow on the dark UI


def test_grey_and_one_colour_wallpapers():
    from jarvis.ui.wallcolors import FALLBACK, palette

    assert palette(Image.new("RGB", (64, 64), (120, 120, 120))) == FALLBACK
    assert palette(Image.new("RGB", (64, 64), (5, 5, 5))) == FALLBACK
    accent, second = palette(Image.new("RGB", (64, 64), (30, 200, 90)))  # all green
    assert accent != second and abs(hue(accent)[0] - hue(second)[0]) > 0.05


def test_signature_changes_with_the_file(tmp_path):
    import os
    import time

    from jarvis.ui.wallcolors import signature

    f = tmp_path / "wall.jpg"
    Image.new("RGB", (8, 8), "red").save(f)
    a = signature(f)
    os.utime(f, (time.time() + 10, time.time() + 10))
    assert signature(f) != a and signature(None) == "" and signature(tmp_path / "gone.jpg") == ""


def test_theme_uses_my_second_colour():
    from jarvis.ui import theme

    theme.apply("Jarvis Blue", "dark", "#00d4ff", "#ff8800")
    assert theme.SECOND == "#ff8800"
    theme.apply("Jarvis Blue", "dark", "#00d4ff", "")
    assert theme.SECOND == theme.THEMES["Jarvis Blue"][1]
    theme.apply("Jarvis Blue", "dark", "#00d4ff", "not a colour")
    assert theme.SECOND == theme.THEMES["Jarvis Blue"][1]


@pytest.fixture
def window(monkeypatch, isolated_home, sandbox):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    qt = QApplication.instance() or QApplication(sys.argv)
    app = create_app(Settings(allowed_roots=[str(sandbox)]))
    w = MainWindow(app, Bridge())
    yield w, app, qt
    w.reminders.close_all()


def test_voice_and_follow_wallpaper(window, monkeypatch):
    from jarvis.config import load_settings
    from jarvis.ui import theme, wallcolors

    w, app, qt = window
    monkeypatch.setattr(wallcolors, "from_wallpaper", lambda: ("#d9bd44", "#5078b2"))
    app.registry.call("jarvis_display", {"action": "wallpaper_theme"}, app.ctx)
    qt.processEvents()
    assert w.accent == "#d9bd44" and theme.SECOND == "#5078b2"
    assert load_settings().theme_second == "#5078b2"
    sig = {"v": "a"}
    monkeypatch.setattr(wallcolors, "signature", lambda p: sig["v"])
    monkeypatch.setattr(wallcolors, "wallpaper_path", lambda: None)
    assert not w._follow_wallpaper()  # off unless asked
    app.settings.follow_wallpaper = True
    monkeypatch.setattr(wallcolors, "from_wallpaper", lambda: ("#44d98a", "#b25078"))
    assert w._follow_wallpaper() and w.accent == "#44d98a"
    assert not w._follow_wallpaper()  # same wallpaper: nothing to do
    sig["v"] = "b"
    monkeypatch.setattr(wallcolors, "from_wallpaper", lambda: ("#ff6655", "#5566ff"))
    assert w._follow_wallpaper() and w.accent == "#ff6655"


def test_settings_colour_buttons(window, monkeypatch):
    from jarvis.ui import wallcolors
    from jarvis.ui.settings_dialog import SettingsDialog

    w, app, qt = window
    app.settings.api_key = "k"
    d = SettingsDialog(app)
    monkeypatch.setattr(wallcolors, "from_wallpaper", lambda: ("#d9bd44", "#5078b2"))
    d._match_wallpaper()
    assert d.color == "#d9bd44" and d.second == "#5078b2"
    d.follow_wall.setChecked(True)
    d._save()
    assert d.result_settings.theme_second == "#5078b2" and d.result_settings.follow_wallpaper
    d._reset_colors()
    assert d.second == "" and not d.follow_wall.isChecked()
    monkeypatch.setattr(wallcolors, "from_wallpaper", lambda: None)
    d._match_wallpaper()
    assert "Couldn't read" in d.color_label.text()
