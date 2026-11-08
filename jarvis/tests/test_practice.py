"""Language practice mode: curriculum, progress store, generator, tutor tools, mode switch, screen."""

import asyncio
import json
import sys
from datetime import date, timedelta
from types import SimpleNamespace as NS

import pytest

from jarvis.english.curriculum import MODULES, day_plan
from jarvis.english.store import EnglishStore

TODAY = date(2026, 11, 9)


# -- curriculum ------------------------------------------------------------------------------
@pytest.mark.parametrize("level", ["beginner", "intermediate", "advanced"])
@pytest.mark.parametrize("focus", list(MODULES))
def test_every_day_plan_is_complete(level, focus):
    for day in (1, 2, 3, 10):
        tasks = day_plan(day, level, focus)
        assert len(tasks) == 5 and len({t.id for t in tasks}) == 5
        kinds = [t.kind for t in tasks]
        assert kinds[:2] == ["mcq", "mcq"] and kinds[2] in ("reading", "listening") and kinds[3] == "speaking"
        for t in tasks:
            if t.kind in ("mcq", "reading", "listening"):
                for q in t.payload["questions"]:
                    assert len(q["options"]) == 4 and 0 <= q["answer"] < 4
            else:
                assert t.payload["prompt"]


def test_plans_are_stable_and_change_by_day():
    assert [t.to_dict() for t in day_plan(3)] == [t.to_dict() for t in day_plan(3)]
    q1 = [q["q"] for q in day_plan(1)[1].payload["questions"]]
    q2 = [q["q"] for q in day_plan(2)[1].payload["questions"]]
    assert q1 != q2
    assert day_plan(1)[2].kind == "reading" and day_plan(2)[2].kind == "listening"


def test_focus_module_drives_speaking():
    tasks = day_plan(1, focus="interview")
    assert tasks[3].module == "interview" and tasks[4].module == "interview"
    assert tasks[3].payload["prompt"] != tasks[4].payload["prompt"]


# -- store -----------------------------------------------------------------------------------
@pytest.fixture
def store(tmp_path):
    return EnglishStore(tmp_path / "english.json")


def test_days_count_practice_days_not_calendar_days(store):
    assert store.day_number(TODAY) == 1
    store.record("d1-vocab", "vocab", 4, 5, today=TODAY)
    assert store.day_number(TODAY) == 1
    assert store.day_number(TODAY + timedelta(days=5)) == 2  # skipped days don't skip content


def test_xp_streak_and_skills(store):
    info = store.record("d1-vocab", "vocab", 4, 5, today=TODAY)
    assert info["xp_gained"] == 4 * 10 + 15 and info["streak"] == 1 and info["first_time"]
    again = store.record("d1-vocab", "vocab", 5, 5, today=TODAY)  # better retry
    assert again["xp_gained"] == 10 and not again["first_time"]
    store.record("d2-grammar", "grammar", 3, 5, today=TODAY + timedelta(days=1))
    assert store.data["streak"] == 2 and store.streak(TODAY + timedelta(days=1)) == 2
    assert store.streak(TODAY + timedelta(days=4)) == 0  # broken
    store.record("d3-grammar", "grammar", 1, 5, today=TODAY + timedelta(days=4))
    assert store.data["streak"] == 1 and store.data["best_streak"] == 2
    s = store.summary(TODAY + timedelta(days=4))
    assert s["skills"]["vocab"] == 80 and s["skills"]["grammar"] == 40 and s["practice_days"] == 3


def test_plan_is_cached_and_rebuilt_on_preference_change(store):
    first = store.plan(TODAY)
    assert [t.id for t in store.plan(TODAY)] == [t.id for t in first]
    store.set_profile(focus="interview")
    assert store.plan(TODAY)[3].module == "interview"
    store.set_profile(level="nonsense", focus="nope")  # ignored
    assert store.profile["focus"] == "interview" and store.profile["level"] == "beginner"


def test_everything_is_remembered(store, tmp_path):
    store.set_profile(native="Urdu", target="English", level="intermediate", focus="routine")
    store.set_mode("practice")
    store.set_current("d1-grammar")
    store.record("d1-vocab", "vocab", 5, 5, today=TODAY)
    store.add_correction("I am agree", "I agree", "agree is a verb")
    again = EnglishStore(tmp_path / "english.json")
    assert again.profile == {"native": "Urdu", "target": "English", "level": "intermediate", "focus": "routine"}
    assert again.mode == "practice" and again.data["current_task"] == "d1-grammar"
    assert again.result("d1-vocab", TODAY)["score"] == 5 and again.data["corrections"][0]["better"] == "I agree"


def test_corrupt_file_starts_fresh(tmp_path):
    (tmp_path / "english.json").write_text("{not json", encoding="utf-8")
    assert EnglishStore(tmp_path / "english.json").profile["target"] == "English"


# -- generator for other languages -------------------------------------------------------------
class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def ask(self, prompt, system=None):
        self.prompts.append(prompt)
        return self.reply


def test_generator_writes_other_languages(store):
    from jarvis.english.generator import make_generator

    q = {"q": "¿Cómo estás?", "options": ["Bien", "Mesa", "Rojo", "Ayer"], "answer": 0, "why": "Bien = fine"}
    llm = FakeLLM("```json\n" + json.dumps({"title": "En el café", "passage": "Hola, quiero un café con leche, por favor.",
                                            "script": "Hola, quiero un café con leche, por favor.",
                                            "questions": [q] * 5}) + "\n```")
    store.set_profile(target="Spanish", native="Hindi")
    tasks = store.plan(TODAY, generator=make_generator(llm))
    assert tasks[0].payload["questions"][0]["q"] == "¿Cómo estás?"
    assert "Spanish" in llm.prompts[0] and "Hindi" in llm.prompts[0]
    assert tasks[3].kind == "speaking"  # speaking prompts are run live by the tutor


def test_generator_falls_back_on_bad_output(store):
    from jarvis.english.generator import make_generator

    store.set_profile(target="French")
    tasks = store.plan(TODAY, generator=make_generator(FakeLLM('{"questions": [{"q": "x"}]}')))
    assert tasks[0].payload.get("fallback") and len(tasks[0].payload["questions"]) == 5


# -- tutor -------------------------------------------------------------------------------------
@pytest.fixture
def app(isolated_home):
    from jarvis.app import create_app
    from jarvis.config import Settings

    return create_app(Settings(api_key="k", sound_effects=False, user_name="Musaddiq"))


def test_tutor_has_only_its_own_tools(app):
    from jarvis.english.tutor import Tutor

    tutor = Tutor(app)
    assert set(tutor.registry.tools) == {"show_mcq", "show_correction", "mark_task_done", "today_plan"}
    for name in ("open_app", "power", "send_whatsapp", "delete_file"):
        assert name not in tutor.registry.tools
    text = tutor.prompt()
    assert "Own language: Hindi" in text and "Learning: English" in text and "d1-speak" in text
    assert "CANNOT control the computer" in text


def test_tutor_tools_update_progress_and_screen(app):
    from jarvis.english.tutor import Tutor

    tutor = Tutor(app)
    seen = []
    app.events.subscribe(lambda k, d: seen.append((k, d)))
    ctx = app.ctx
    assert "error" in tutor.registry.call("show_mcq", {"question": "?", "options": ["a"], "answer": 3}, ctx)
    tutor.registry.call("show_mcq", {"question": "Pick", "options": ["a", "b"], "answer": 1, "why": "b"}, ctx)
    tutor.registry.call("show_correction", {"said": "I am agree", "better": "I agree", "tip": "no 'am'"}, ctx)
    out = tutor.registry.call("mark_task_done", {"task_id": "d1-speak", "score": 8}, ctx)
    assert "+95 XP" in out["result"]
    assert "error" in tutor.registry.call("mark_task_done", {"task_id": "nope", "score": 1}, ctx)
    kinds = [k for k, _ in seen]
    assert {"practice_mcq", "practice_correction", "practice_progress"} <= set(kinds)
    plan = tutor.registry.call("today_plan", {}, ctx)
    assert any(t["id"] == "d1-speak" and t["done"] for t in plan["tasks"])


def test_declarations_are_valid_for_gemini(app):
    from google.genai import types

    from jarvis.english.tutor import Tutor

    for decl in Tutor(app).registry.declarations():
        types.FunctionDeclaration(**decl)


# -- live session mode switch --------------------------------------------------------------------
def test_mode_switch_swaps_prompt_tools_and_handles(app):
    from jarvis.live import LiveAssistant

    a = LiveAssistant(app)
    a.resume_handle = "agent-handle"
    a.set_mode("practice")
    cfg = a._config()
    names = {f.name for f in cfg.tools[0].function_declarations}
    assert names == {"show_mcq", "show_correction", "mark_task_done", "today_plan"}
    assert "LANGUAGE PRACTICE MODE" in cfg.system_instruction
    assert a.resume_handle is None
    a.set_mode("agent")
    assert a.resume_handle == "agent-handle" and "open_app" in {f.name for f in a._config().tools[0].function_declarations}


def test_agent_commands_are_refused_in_practice(app):
    from jarvis.live import LiveAssistant

    a = LiveAssistant(app)
    a.set_mode("practice")
    sent = []

    class Session:
        async def send_tool_response(self, function_responses):
            sent.extend(function_responses)

    call = NS(function_calls=[NS(id="1", name="power", args={"action": "shutdown"})])
    asyncio.run(a._handle_tools(Session(), call))
    assert "Practice mode is on" in sent[0].response["error"]


# -- the screen ------------------------------------------------------------------------------------
@pytest.fixture
def window(monkeypatch, isolated_home):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    qt = QApplication.instance() or QApplication(sys.argv)
    app = create_app(Settings())
    w = MainWindow(app, Bridge())
    w.qt = qt
    return w, app


def test_button_switches_modes_and_is_remembered(window):
    w, app = window
    w.practice_btn.click()
    assert w.practicing and w.stack.currentWidget() is w.practice and w.chips.isHidden()
    assert app.english.mode == "practice" and "TUTOR" in w.title.text()
    w.practice_btn.click()
    assert not w.practicing and w.stack.currentWidget() is w.chat and app.english.mode == "agent"


def test_quiz_on_screen_records_progress(window):
    w, app = window
    w.set_practice(True)
    task = app.english.plan()[0]
    w.practice.open_task(task)
    from jarvis.ui.practice import QuizView

    view = w.practice.area.findChildren(QuizView)[-1]
    for q in task.payload["questions"]:
        view.answer(q["answer"])  # all correct
        view._advance()
    r = app.english.result(task.id)
    assert r["score"] == 5 and r["total"] == 5
    assert "Excellent" in " ".join(lbl.text() for lbl in w.practice.area.findChildren(type(w.title)))


def test_tutor_events_show_on_screen(window):
    w, app = window
    w.set_practice(True)
    app.events.emit("practice_mcq", question="Pick b", options=["a", "b"], answer=1, why="b is right")
    w.qt.processEvents()
    texts = " ".join(lbl.text() for lbl in w.practice.findChildren(type(w.title)))
    assert "Quick quiz" in texts and "Pick b" in texts
    app.events.emit("practice_correction", said="I am agree", better="I agree", tip="agree is a verb")
    w.qt.processEvents()
    texts = " ".join(lbl.text() for lbl in w.practice.findChildren(type(w.title)))
    assert "I agree" in texts and "agree is a verb" in texts


def test_language_choice_is_saved(window):
    w, app = window
    w.set_practice(True)  # the practice screen only exists once the mode is used
    w.practice.native.setCurrentText("Urdu")
    w.practice.level.setCurrentIndex(w.practice.level.findData("advanced"))
    assert app.english.profile["native"] == "Urdu" and app.english.profile["level"] == "advanced"
