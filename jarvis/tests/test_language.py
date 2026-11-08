"""Language choice: what Jarvis speaks, and what reminders use."""

import json
import sys
import time

import pytest

from jarvis import lang
from jarvis.config import Settings


@pytest.mark.parametrize("text, expected", [
    ("Mere liye reminder set kar do, mummy ko call karna hai", "hinglish"),
    ("kal subah 9 baje yaad dilana", "hinglish"),
    ("Remind me to do the dishes at 9", "en"),
    ("What's the weather like today?", "en"),
    ("मुझे कल सुबह याद दिलाना", "hi"),
    ("مجھے کل صبح یاد دلانا", "ur"),
    ("ذكرني غدا صباحا", "ar"),
    ("நாளை காலை நினைவூட்டு", "ta"),
    ("రేపు ఉదయం గుర్తు చేయి", "te"),
    ("ok", None),
    ("👍👍", None),
])
def test_detect(text, expected):
    assert lang.detect(text) == expected


def test_find_names_and_aliases():
    assert lang.find("Hindi") == "hi" and lang.find("हिंदी") == "hi" and lang.find("in hinglish please") == "hinglish"
    assert lang.find("angrezi") == "en" and lang.find("ta") == "ta" and lang.find("auto") == "auto"
    assert lang.find("klingon") is None and lang.find("") is None


class State(dict):
    def set(self, k, v):
        self[k] = v


def test_resolving_choices():
    st = State(user_lang="hinglish")
    assert lang.reply_language(Settings(), st) == "hinglish"  # auto follows the user
    assert lang.reply_language(Settings(), State()) == "en"  # nothing known yet
    assert lang.reply_language(Settings(reply_language="hi"), st) == "hi"
    assert lang.reminder_language(Settings(reply_language="hi"), st) == "hi"  # same as Jarvis
    assert lang.reminder_language(Settings(reply_language="hi", reminder_language="en"), st) == "en"
    assert lang.reminder_language(Settings(reply_language="hi", reminder_language="auto"), st) == "hinglish"
    assert lang.reply_language(Settings(reply_language="xx"), st) == "en"  # a broken config value


def test_prompt_rules():
    auto = lang.prompt_rules(Settings())
    assert "language the user speaks" in auto and "Reminders" not in auto
    fixed = lang.prompt_rules(Settings(reply_language="hi", reminder_language="en"))
    assert "Always reply in Hindi" in fixed and "Reminders are in English" in fixed and "set_language" in fixed
    follow = lang.prompt_rules(Settings(reply_language="en", reminder_language="auto"))
    assert "Reminders follow the language the user speaks" in follow


def test_every_language_has_every_word():
    for code, words in lang.STRINGS.items():
        assert set(words) == set(lang._KEYS), code
        words["in_min"].format(n=5)
        words["in_hm"].format(h=1, m=5)
        words["in_days"].format(d=2)
        words["minutes"].format(n=5)
    assert set(lang.STRINGS) == set(lang.LANGUAGES)


def test_noticing_the_users_language(isolated_home):
    from jarvis.app import create_app

    app = create_app(Settings())
    app.events.emit("user", text="Bhai mera kal ka plan kya hai?")
    assert app.state.get("user_lang") == "hinglish"
    app.events.emit("user", text="ok")  # too short: keep the last guess
    assert app.state.get("user_lang") == "hinglish"
    app.events.emit("user", text="What's on my calendar tomorrow?")
    assert app.state.get("user_lang") == "en"


def test_prompt_follows_settings(isolated_home):
    from jarvis.app import create_app

    app = create_app(Settings(reply_language="ta"))
    assert "Always reply in Tamil" in app.prompt()


# -- the voice command --------------------------------------------------------------------------
@pytest.fixture
def lctx(ctx, isolated_home):
    from jarvis.events import Events

    ctx.events = Events()
    ctx.seen = []
    ctx.events.subscribe(lambda k, d: ctx.seen.append((k, d)))
    return ctx


def saved():
    from jarvis.config import settings_path

    return json.loads(settings_path().read_text(encoding="utf-8"))


def test_set_language_for_jarvis(registry, lctx):
    out = registry.call("set_language", {"language": "Hindi"}, lctx)
    assert "Hindi" in out["result"] and lctx.settings.reply_language == "hi"
    assert saved()["reply_language"] == "hi"
    assert lctx.seen[-1] == ("language", {"reply": "hi", "reminders": "same"})


def test_set_language_for_reminders_and_both(registry, lctx):
    registry.call("set_language", {"language": "english", "for": "reminders"}, lctx)
    assert lctx.settings.reminder_language == "en" and lctx.settings.reply_language == "auto"
    registry.call("set_language", {"language": "hinglish", "for": "both"}, lctx)
    assert (lctx.settings.reply_language, lctx.settings.reminder_language) == ("hinglish", "same")
    registry.call("set_language", {"language": "auto"}, lctx)
    assert lctx.settings.reply_language == "auto"


def test_set_language_errors(registry, lctx):
    assert "I can use" in registry.call("set_language", {"language": "klingon"}, lctx)["error"]
    assert "error" in registry.call("set_language", {"language": "same"}, lctx)
    assert lctx.settings.reply_language == "auto"


def test_choices_survive_a_restart(isolated_home):
    from jarvis.config import load_settings, save_settings

    save_settings(Settings(reply_language="ur", reminder_language="en"))
    s = load_settings()
    assert (s.reply_language, s.reminder_language) == ("ur", "en")


# -- pop-ups and settings window --------------------------------------------------------------
@pytest.fixture
def qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


def test_popup_words_follow_the_language(qt):
    from jarvis.ui.reminder_popup import ReminderPopups, until

    rp = ReminderPopups(lambda: "#00d4ff", on_done=print, on_snooze=print, on_cancel=print, on_open=print,
                        language=lambda: "hi")
    card = rp.show("due", 1, "मम्मी को कॉल करना है", time.time())
    assert card.head.text().startswith("रिमाइंडर") and card.buttons[0].text() == "✓  हो गया"
    assert card.sub.text() == "आपका काम अब करना है"
    rp.language = lambda: "hinglish"
    card = rp.show("set", 2, "Gym", time.time() + 25 * 60, repeat="daily")
    assert "ROZ" in card.head.text() and "25 min me" in card.sub.text() and card.buttons[0].text() == "Reminder hatao"
    rp.language = lambda: 1 / 0  # a broken lookup still shows the card, in English
    assert rp.show("due", 3, "x", time.time()).buttons[0].text() == "✓  Done"
    now = time.time()
    assert until(now + 86400 + 60, now, "en") == "in 1 day" and until(now + 3 * 3600, now, "hi") == "3 घंटे में"
    rp.close_all()


def test_window_uses_the_reminder_language(qt, isolated_home, sandbox):
    from jarvis.app import create_app
    from jarvis.ui.window import Bridge, MainWindow

    app = create_app(Settings(allowed_roots=[str(sandbox)], reminder_language="auto"))
    w = MainWindow(app, Bridge())
    app.events.emit("user", text="kal subah yaad dilana bhai")
    app.registry.call("set_reminder", {"message": "Doodh lana", "minutes": 30}, app.ctx)
    qt.processEvents()
    assert w.reminders.popups[-1].lang == "hinglish"
    app.registry.call("cancel_reminder", {"id": w.reminders.popups[-1].rid}, app.ctx)
    w.reminders.close_all()


def test_settings_dialog_pickers(qt, isolated_home):
    from jarvis.app import create_app
    from jarvis.ui.settings_dialog import SettingsDialog

    app = create_app(Settings(api_key="k", reply_language="hi", reminder_language="same"))
    d = SettingsDialog(app)
    assert d.reply_lang.currentData() == "hi" and d.reminder_lang.currentData() == "same"
    assert d.reply_lang.count() == len(lang.LANGUAGES) + 1 and d.reminder_lang.count() == len(lang.LANGUAGES) + 2
    d.reply_lang.setCurrentIndex(d.reply_lang.findData("hinglish"))
    d.reminder_lang.setCurrentIndex(d.reminder_lang.findData("en"))
    d._save()
    assert (d.result_settings.reply_language, d.result_settings.reminder_language) == ("hinglish", "en")
