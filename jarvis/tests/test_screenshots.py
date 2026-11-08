"""take_screenshot / open_file / latest_screenshot and the log file."""

import pytest
from PIL import Image


@pytest.fixture
def shots(tmp_path, monkeypatch):
    import jarvis.tools.files as files
    import jarvis.tools.vision as vision

    folder = tmp_path / "sandbox" / "Jarvis Screenshots"
    monkeypatch.setattr(vision, "SCREENSHOT_DIR", folder)
    monkeypatch.setattr(vision, "grab_screen", lambda monitor=1: Image.new("RGB", (64, 40), "teal"))
    opened = []
    monkeypatch.setattr(files, "open_path", opened.append)
    return folder, opened


def test_take_screenshot_saves_opens_and_shows(registry, ctx, shots):
    folder, opened = shots
    seen = []
    ctx.attach_image = seen.append
    out = registry.call("take_screenshot", {}, ctx)
    saved = list(folder.glob("*.png"))
    assert len(saved) == 1 and out["image"] == str(saved[0])
    assert opened == [saved[0]]
    assert seen and seen[0][:2] == b"\xff\xd8"  # the model also sees it (JPEG)
    assert "opened" in out["result"]


def test_take_screenshot_without_opening(registry, ctx, shots):
    folder, opened = shots
    out = registry.call("take_screenshot", {"open": False}, ctx)
    assert "error" not in out and opened == []


def test_latest_screenshot(registry, ctx, shots):
    assert "error" in registry.call("latest_screenshot", {}, ctx)
    registry.call("take_screenshot", {"open": False}, ctx)
    out = registry.call("latest_screenshot", {}, ctx)
    assert out["path"].endswith(".png") and out["image"] == out["path"]


def test_open_file(registry, ctx, sandbox, shots):
    _, opened = shots
    pic = sandbox / "cat.png"
    Image.new("RGB", (4, 4)).save(pic)
    out = registry.call("open_file", {"path": str(pic)}, ctx)
    assert opened == [pic] and out["image"] == str(pic)
    assert "error" in registry.call("open_file", {"path": str(sandbox / "missing.pdf")}, ctx)


def test_open_file_refuses_programs_and_outside_paths(registry, ctx, sandbox, shots, tmp_path):
    _, opened = shots
    bad = sandbox / "run.bat"
    bad.write_text("echo hi")
    assert "error" in registry.call("open_file", {"path": str(bad)}, ctx)
    outside = tmp_path / "secret.txt"
    outside.write_text("x")
    assert "error" in registry.call("open_file", {"path": str(outside)}, ctx)
    assert opened == []


def test_look_at_screen_keeps_a_copy_for_the_window(registry, ctx, monkeypatch, isolated_home):
    import jarvis.tools.vision as vision

    monkeypatch.setattr(vision, "screenshot_jpeg", lambda monitor=1: b"\xff\xd8JPEG")
    ctx.attach_image = lambda jpeg: None
    out = registry.call("look_at_screen", {}, ctx)
    assert out["image"].endswith(".jpg") and (isolated_home / "cache").is_dir()


def test_prompt_knows_about_screenshots(ctx):
    from jarvis.app import system_prompt

    text = system_prompt(ctx.settings, ctx.memory)
    assert "take_screenshot" in text and "open_file" in text


def test_logging_writes_a_file(isolated_home):
    import logging

    from jarvis.logs import setup_logging

    path = setup_logging()
    logging.getLogger("jarvis.test").warning("hello log")
    for h in logging.getLogger("jarvis").handlers:
        h.flush()
    assert path is not None and "hello log" in path.read_text(encoding="utf-8")
