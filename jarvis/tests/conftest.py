import pytest

from jarvis.config import Settings
from jarvis.confirm import ConfirmGate
from jarvis.memory import MemoryStore
from jarvis.tools import ToolContext, build_registry
from jarvis.undo import UndoStack


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return tmp_path / "home"


@pytest.fixture
def sandbox(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    return root


@pytest.fixture
def answers():
    """List of yes/no answers the fake user gives to confirmations."""
    return []


@pytest.fixture
def ctx(tmp_path, sandbox, answers):
    settings = Settings(allowed_roots=[str(sandbox)])
    gate = ConfirmGate(lambda q: answers.pop(0) if answers else False, timeout=2)
    messages = []
    c = ToolContext(settings, MemoryStore(tmp_path / "mem.json"), UndoStack(), gate, messages.append)
    c.messages = messages
    return c


@pytest.fixture(scope="session")
def registry():
    return build_registry()


_KEEP: list = []  # the last test's windows: hidden, timers off, deleted one test later


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Before a test's fixtures let go of its windows, hold on to them.

    Destroying a Qt window while its animations run can leave Qt's animation clock stuck (later animations
    never move) or crash a later test. So finished tests' windows are hidden and their timers stopped
    instead; their short animations just run out.
    """
    import sys

    qt = sys.modules.get("PyQt6.QtWidgets")
    app = qt.QApplication.instance() if qt else None
    if app is not None:
        from PyQt6.QtCore import QAbstractAnimation, QCoreApplication, QEvent, QTimer

        old = list(_KEEP)  # windows from the test before: hidden and still for a whole test, safe to delete now
        _KEEP.clear()
        for w in old:
            try:
                w.deleteLater()
            except RuntimeError:
                pass
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        for w in app.topLevelWidgets():
            if any(w is k for k in _KEEP):
                continue
            _KEEP.append(w)
            try:
                for t in w.findChildren(QTimer):
                    t.stop()
                for a in w.findChildren(QAbstractAnimation):  # safe: the window stays alive
                    a.stop()
                w.hide()
            except RuntimeError:
                pass
    yield
