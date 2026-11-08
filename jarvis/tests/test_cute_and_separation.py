"""Cute moments, the instructor look, and keeping practice mode completely separate."""

import subprocess
import sys
import time

import pytest


@pytest.fixture
def qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


# -- the orb --------------------------------------------------------------------------------
def test_transform_morphs_in_and_back(qt):
    from jarvis.ui.orb import Orb

    orb = Orb("#00d4ff")
    orb.transform("star", seconds=0.2)
    assert orb.cute_kind == "star" and len(orb.sparkles) == 14
    for _ in range(30):
        orb._last -= 0.02
        orb._tick()
    assert orb.cute > 0.5  # on its way to the cute face
    orb.cute_until = time.monotonic() - 1  # time's up
    for _ in range(60):
        orb._last -= 0.02
        orb._tick()
    assert orb.cute == 0.0  # back to the AI core


def test_unknown_kind_is_happy_and_face_off_stays_ai(qt):
    from jarvis.ui.orb import Orb

    orb = Orb("#00d4ff", face=False)
    orb.transform("weird")
    assert orb.cute_kind == "happy"
    orb._last -= 0.05
    orb._tick()
    assert orb.cute == 0.0  # with the face turned off there is no cute face


@pytest.mark.parametrize("kind", ["love", "happy", "wink", "star"])
@pytest.mark.parametrize("persona", ["", "tutor"])
def test_every_look_paints(qt, kind, persona):
    from jarvis.ui.orb import Orb

    orb = Orb("#00d4ff")
    orb.resize(300, 330)
    orb.boot = 1.0
    orb.set_persona(persona)
    orb.transform(kind)
    orb.cute = 0.5
    orb.point()
    orb.letters = [[0.3, 0.1, "A"]]
    assert not orb.grab().toImage().isNull()


def test_surprise_happens_only_when_idle(qt):
    from jarvis.ui.orb import Orb

    orb = Orb("#00d4ff")
    orb.set_state("speaking")
    orb._next_surprise = 0
    orb._tick()
    assert orb.cute_until == 0.0
    orb.set_state("listening")
    orb._next_surprise = 0
    orb._tick()
    assert orb.cute_until > time.monotonic()


# -- window reactions --------------------------------------------------------------------------
@pytest.fixture
def window(qt, isolated_home):
    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    app = create_app(Settings())
    return MainWindow(app, Bridge()), app, qt


def test_praise_and_celebration_turn_cute(window):
    w, app, qt = window
    w._react_to_user("wow, zabardast!")
    assert w.big_orb.cute_kind == "happy"
    w.big_orb.cute_until = 0
    app.events.emit("assistant", text="Congratulations on the new job!")
    qt.processEvents()
    assert w.big_orb.cute_kind == "star" and w.big_orb.cute_until > time.monotonic()


def test_practice_reactions(window):
    w, app, qt = window
    w.set_practice(True)
    assert w.big_orb.persona == "tutor"
    app.events.emit("practice_answer", correct=True)
    assert w.big_orb.cute_kind == "happy"
    app.events.emit("practice_answer", correct=False)
    assert w.big_orb.emotion == "sad"
    app.events.emit("practice_correction", said="I am agree", better="I agree", tip="")
    assert w.big_orb.point_t == 1.0
    app.events.emit("practice_progress", task_id="d1-vocab", score=5.0, total=5.0, xp_gained=65, streak=1)
    assert w.big_orb.cute_kind == "star"
    w.set_practice(False)
    assert w.big_orb.persona == ""


# -- separation -----------------------------------------------------------------------------------
def test_practice_code_does_not_load_until_turned_on(isolated_home):
    """A normal start must not import or create anything of the practice module."""
    code = (
        "import os, sys\n"
        "os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
        "from PyQt6.QtWidgets import QApplication\n"
        "qt = QApplication(sys.argv)\n"
        "from jarvis.app import create_app\n"
        "from jarvis.config import Settings, home_dir\n"
        "from jarvis.ui.window import Bridge, MainWindow\n"
        "app = create_app(Settings()); w = MainWindow(app, Bridge())\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('jarvis.english') or m == 'jarvis.ui.practice')\n"
        "print('LOADED', loaded, (home_dir() / 'english.json').exists())\n"
        "w.set_practice(True)\n"
        "print('AFTER', 'jarvis.ui.practice' in sys.modules, (home_dir() / 'english.json').exists())\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60).stdout
    assert "LOADED [] False" in out
    assert "AFTER True True" in out


def test_practice_flag_lives_in_jarvis_state(window):
    w, app, qt = window
    assert app.practice_on is False and w.practice is None
    w.set_practice(True)
    assert app.state.get("practice_mode") is True and app.practice_on
    w.set_practice(False)
    assert app.state.get("practice_mode") is False


def test_agent_session_never_sees_tutor_tools(isolated_home):
    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.live import LiveAssistant

    a = LiveAssistant(create_app(Settings(api_key="k", sound_effects=False)))
    names = {f.name for f in a._config().tools[0].function_declarations}
    assert not names & {"show_mcq", "show_correction", "mark_task_done", "today_plan"}
    assert a.practice is None and "LANGUAGE PRACTICE" not in a._config().system_instruction
