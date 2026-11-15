"""The test suite's own window clean-up (conftest.py) must survive popups that Qt creates by itself."""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

POPUP_TESTS = '''
import sys
import weakref

import pytest


@pytest.fixture
def qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


WINDOWS = []
APPS = []


def test_window_with_an_open_drop_down(qt):
    APPS.append(weakref.ref(qt))  # weak: this test must not be what keeps the app alive
    from PyQt6.QtWidgets import QComboBox, QVBoxLayout, QWidget

    w = QWidget()
    WINDOWS.append(w)
    box = QComboBox(w)
    box.addItems(["one", "two"])
    QVBoxLayout(w).addWidget(box)
    w.show()
    box.showPopup()  # Qt makes the drop-down as a separate top-level window
    qt.processEvents()


def test_next(qt):
    from PyQt6 import sip

    assert APPS[0]() is qt and not sip.isdeleted(WINDOWS[0])  # same app, and the parked window is still there
    WINDOWS.pop().deleteLater()  # the app's own code closes and deletes a window a test later
    qt.processEvents()  # its drop-down goes with it, behind sip's back


def test_and_the_one_after(qt):
    qt.processEvents()
'''


def test_cleanup_skips_popups_qt_owns(tmp_path):
    shutil.copy(ROOT / "tests" / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_popup.py").write_text(POPUP_TESTS, encoding="utf-8")
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
                         cwd=tmp_path, capture_output=True, text=True, timeout=120,
                         env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    assert "3 passed" in run.stdout
