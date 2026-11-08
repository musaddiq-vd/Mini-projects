"""More than one monitor: where pop-ups, the widget and the mini window appear."""

import sys
import time

import pytest
from PyQt6.QtCore import QPoint, QRect


class FakeScreen:
    def __init__(self, name, rect):
        self._name, self._rect = name, rect

    def name(self):
        return self._name

    def availableGeometry(self):  # noqa: N802
        return self._rect

    def geometry(self):
        return self._rect


LEFT = FakeScreen("DISPLAY1", QRect(0, 0, 1920, 1040))
RIGHT = FakeScreen("DISPLAY2", QRect(1920, 0, 2560, 1400))


@pytest.fixture
def qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def two_screens(monkeypatch, qt):
    from jarvis.ui import screens

    monkeypatch.setattr(screens, "screens", lambda: [LEFT, RIGHT])
    monkeypatch.setattr(screens, "main_screen", lambda: LEFT)
    monkeypatch.setattr(screens, "at_point", lambda p: RIGHT if p.x() >= 1920 else LEFT)
    return screens


def test_choosing_a_screen(two_screens, monkeypatch):
    s = two_screens
    assert s.by_name("DISPLAY2") is RIGHT and s.by_name("") is LEFT and s.by_name("unplugged") is LEFT
    monkeypatch.setattr(s.QCursor, "pos", staticmethod(lambda: QPoint(2500, 300)))
    assert s.for_popups("mouse") is RIGHT and s.for_popups("main") is LEFT

    class Win:
        def isVisible(self):
            return True

        def screen(self):
            return LEFT

    assert s.for_popups("jarvis", Win()) is LEFT
    assert s.visible_anywhere(QPoint(3000, 500)) and not s.visible_anywhere(QPoint(9000, 500))
    assert "main" in s.label(LEFT, 0) and "2560" in s.label(RIGHT, 1)


def test_popup_stack_stays_on_one_screen(qt):
    from jarvis.ui.reminder_popup import ReminderPopups

    where = {"screen": RIGHT}
    rp = ReminderPopups(lambda: "#00d4ff", on_done=print, on_snooze=print, on_cancel=print, on_open=print,
                        screen=lambda: where["screen"])
    a = rp.show("due", 1, "One", time.time())
    where["screen"] = LEFT  # the mouse moved to the other monitor
    b = rp.show("due", 2, "Two", time.time())
    assert a.pos().x() > 1920 and b.pos().x() > 1920  # a stack doesn't split across monitors
    rp.close_all()
    c = rp.show("due", 3, "Three", time.time())
    assert c.pos().x() < 1920  # a new stack follows the mouse
    rp.close_all()


def test_widget_goes_to_its_monitor_and_falls_back(two_screens, qt, isolated_home):
    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.desktop_widget import DesktopWidget

    d = DesktopWidget(create_app(Settings()), "#00d4ff", screen_name="DISPLAY2")
    d.place()
    assert d.x() > 1920 and d.y() == 28
    d.place("bottom_left", "unplugged-monitor")
    assert d.x() == 28 and d.y() + d.height() <= 1040  # back on the main screen
    d.close()
