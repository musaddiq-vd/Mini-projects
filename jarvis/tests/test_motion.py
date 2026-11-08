"""Voice control of Jarvis' own window and scrolling the screen."""

import sys
import time
from types import SimpleNamespace

import pytest


@pytest.fixture
def qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


def settle(qt, ms):
    end = time.time() + ms / 1000
    while time.time() < end:
        qt.processEvents()
        time.sleep(0.005)


def finish(qt, m, limit=3.0):
    """Wait until the mover's animation has ended (slow machines need longer)."""
    from PyQt6.QtCore import QAbstractAnimation

    end = time.time() + limit
    while time.time() < end and m._anim is not None and m._anim.state() == QAbstractAnimation.State.Running:
        settle(qt, 20)
        if m._anim is not None and m._anim.currentTime() == 0 and time.time() > end - limit + 0.5:
            break  # pytest's Qt animation clock can stall after many windows; jump to the end below
    if m._anim is not None and m._anim.state() == QAbstractAnimation.State.Running:
        m._anim.setCurrentTime(m._anim.duration())  # lands exactly where the animation would have
    settle(qt, 20)


@pytest.fixture
def win(qt):
    from PyQt6.QtWidgets import QWidget

    w = QWidget()
    w.resize(400, 300)
    w.show()
    area = w.screen().availableGeometry()
    w.move(area.center().x() - 200, area.center().y() - 150)
    settle(qt, 30)
    yield w
    w.close()


def test_nudge_and_edges(qt, win):
    from jarvis.ui.mover import WindowMover

    m = WindowMover(lambda: win)
    y0 = win.y()
    m.nudge("down", "little")
    finish(qt, m)
    assert win.y() == y0 + 70
    m.nudge("down", "lot")
    m.nudge("down", "lot")
    m.nudge("down", "lot")
    finish(qt, m)
    area = win.screen().availableGeometry()
    assert win.frameGeometry().bottom() <= area.bottom() + 1  # never off screen
    assert m.nudge("sideways") == "unknown direction"


def test_place_corners_and_halves(qt, win):
    from jarvis.ui.mover import WindowMover

    m = WindowMover(lambda: win)
    area = win.screen().availableGeometry()
    m.place("top_left")
    finish(qt, m)
    assert (win.x(), win.y()) == (area.left() + 12, area.top() + 12)
    m.place("bottom_right")
    finish(qt, m)
    assert win.frameGeometry().right() == area.right() - 12
    m.place("right_half")
    settle(qt, 50)
    f = win.frameGeometry()
    assert f.left() == area.left() + area.width() // 2 and f.height() == area.height()
    assert m.place("moon") == "unknown place"


def test_glide_until_stop_and_edge(qt, win):
    from jarvis.ui.mover import WindowMover

    m = WindowMover(lambda: win)
    x0 = win.x()
    m.glide("right", "fast")
    end = time.time() + 2
    while win.x() == x0 and time.time() < end:  # the first frames can be slow on a busy machine
        settle(qt, 20)
    assert m.gliding and win.x() > x0
    m.stop()
    x1 = win.x()
    settle(qt, 100)
    assert not m.gliding and win.x() == x1
    m.glide("left", "fast")
    end = time.time() + 6
    while m.gliding and time.time() < end:
        settle(qt, 30)
    assert not m.gliding and win.x() == win.screen().availableGeometry().left()  # stops at the edge


def test_resize_and_maximised(qt, win):
    from jarvis.ui.mover import WindowMover

    m = WindowMover(lambda: win)
    w0 = win.width()
    m.resize(True)
    assert win.width() > w0
    m.resize(False)
    m.resize(False)
    assert win.width() < w0
    win.setFixedSize(200, 200)
    assert m.resize(True) == "fixed"


# -- the tool and the main window ------------------------------------------------------------
def test_display_tool_checks_and_sends(registry, ctx):
    from jarvis.events import Events

    ctx.events = Events()
    seen = []
    ctx.events.subscribe(lambda k, d: seen.append(d) if k == "ui_command" else None)
    assert "error" in registry.call("jarvis_display", {"action": "move"}, ctx)
    assert "error" in registry.call("jarvis_display", {"action": "place", "place": "moon"}, ctx)
    out = registry.call("jarvis_display", {"action": "glide", "direction": "down", "speed": "slow"}, ctx)
    assert "stop" in out["result"] and seen[-1]["action"] == "glide" and seen[-1]["speed"] == "slow"


def test_main_window_moves_by_voice(qt, isolated_home, sandbox):
    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    app = create_app(Settings(allowed_roots=[str(sandbox)]))
    w = MainWindow(app, Bridge())
    w.showNormal()
    w.setMinimumSize(0, 0)
    w.resize(500, 400)  # the test screen is small
    settle(qt, 50)
    app.registry.call("jarvis_display", {"action": "place", "place": "top_left"}, app.ctx)
    finish(qt, w.mover)
    area = w.screen().availableGeometry()
    fits = w.frameGeometry().width() + 24 <= area.width()
    assert w.y() == area.top() + 12 and w.x() == (area.left() + 12 if fits else area.left())
    app.registry.call("jarvis_display", {"action": "glide", "direction": "down"}, app.ctx)
    settle(qt, 60)
    assert w.mover.gliding
    app.registry.call("stop_moving", {}, app.ctx)
    assert not w.mover.gliding
    app.registry.call("jarvis_display", {"action": "minimize"}, app.ctx)
    settle(qt, 50)
    assert w.isMinimized()
    w.reminders.close_all()


# -- scrolling the screen --------------------------------------------------------------------
@pytest.fixture
def fake_screen(monkeypatch):
    import os

    from jarvis.tools import _win, motion

    pressed = []
    reader = SimpleNamespace(hwnd=11, pid=4242, app_name="chrome", title="Docs")
    me = SimpleNamespace(hwnd=22, pid=os.getpid(), app_name="python", title="Jarvis")
    state = {"fg": me}
    monkeypatch.setattr(motion, "OS", "Windows")
    monkeypatch.setattr(_win, "foreground", lambda: state["fg"])
    monkeypatch.setattr(_win, "list_windows", lambda: [me, reader])
    monkeypatch.setattr(_win, "focus", lambda h: state.update(fg=reader) or True)
    monkeypatch.setattr(motion, "_press", lambda key, times=1: pressed.extend([key] * times))
    yield SimpleNamespace(pressed=pressed, state=state, reader=reader)
    motion.stop_scrolling()


def test_scroll_the_window_behind_jarvis(registry, ctx, fake_screen):
    out = registry.call("scroll_screen", {"direction": "down"}, ctx)
    assert "chrome" in out["result"] and fake_screen.pressed == ["down"] * 3
    registry.call("scroll_screen", {"direction": "up", "amount": "page"}, ctx)
    registry.call("scroll_screen", {"direction": "top"}, ctx)
    assert fake_screen.pressed[-2:] == ["pageup", "home"]
    assert "error" in registry.call("scroll_screen", {"direction": "diagonal"}, ctx)


def test_keep_scrolling_until_stop(registry, ctx, fake_screen):
    from jarvis.tools import motion

    out = registry.call("scroll_screen", {"direction": "down", "keep_going": True, "speed": "fast"}, ctx)
    assert "stop" in out["result"]
    time.sleep(0.3)
    assert motion.scrolling() and len(fake_screen.pressed) >= 3
    assert registry.call("stop_moving", {}, ctx)["result"] == "Stopped."
    time.sleep(0.15)
    n = len(fake_screen.pressed)
    time.sleep(0.2)
    assert not motion.scrolling() and len(fake_screen.pressed) == n


def test_scrolling_stops_when_the_user_switches_window(registry, ctx, fake_screen):
    from jarvis.tools import motion

    registry.call("scroll_screen", {"direction": "down", "keep_going": True, "speed": "fast"}, ctx)
    time.sleep(0.15)
    fake_screen.state["fg"] = SimpleNamespace(hwnd=99, pid=1, app_name="word", title="x")
    time.sleep(0.25)
    assert not motion.scrolling()
