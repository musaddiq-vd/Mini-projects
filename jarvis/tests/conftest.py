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
