"""Offline mode: simple commands without internet."""

import time

import pytest

from jarvis.offline import addressed, parse, reply


@pytest.mark.parametrize("text, tool, args", [
    ("5 minute ka timer lagao", "set_timer", {"minutes": 5.0}),
    ("Jarvis, set a timer for five minutes", "set_timer", {"minutes": 5.0}),
    ("timer for 30 seconds for eggs", "set_timer", {"seconds": 30.0, "label": "eggs"}),
    ("set a timer for 1 hour 10 minutes", "set_timer", {"minutes": 70.0}),
    ("timer kitna bacha hai", "list_timers", {}),
    ("remind me in 10 minutes to call mom", "set_reminder", {"message": "call mom", "minutes": 10.0}),
    ("remind me at 6:30 pm to take medicine", "set_reminder", {"message": "take medicine", "at": "6:30 pm"}),
    ("remind me to drink water in 20 minutes", "set_reminder", {"message": "drink water", "minutes": 20.0}),
    ("10 minute baad yaad dilana ki chai bani hai", "set_reminder", {"message": "chai bani hai", "minutes": 10.0}),
    ("shaam 6 baje yaad dilana ki mummy ko call karna hai", "set_reminder",
     {"message": "mummy ko call karna hai", "at": "shaam 6"}),
    ("milk aur bread list me daal do", "add_note", {"text": "milk aur bread", "todo": True}),
    ("add eggs to my list", "add_note", {"text": "eggs"}),
    ("note karo gate code 4321 hai", "add_note", {"text": "gate code 4321 hai", "todo": False}),
    ("meri list batao", "list_notes", {}),
    ("my reminders", "list_reminders", {}),
    ("volume badhao", "volume", {"action": "up"}),
    ("volume kam karo", "volume", {"action": "down"}),
    ("what time is it", "get_time", {}),
    ("open notepad", "open_app", {"name": "notepad"}),
    ("chrome kholo", "open_app", {"name": "chrome"}),
])
def test_parse(text, tool, args):
    cmd = parse(text)
    assert cmd is not None and cmd.tool == tool
    for k, v in args.items():
        assert cmd.args[k] == v, (k, cmd.args)


@pytest.mark.parametrize("text", ["what's the weather", "tell me a joke", "hello", "", "open youtube.com website",
                                  "cancel the timer", "why is the sky blue"])
def test_not_offline_commands(text):
    assert parse(text) is None


def test_only_commands_that_start_with_the_name():
    assert addressed("Jarvis set a timer for five minutes")
    assert addressed("hey Jarvis what time is it")
    assert not addressed("I told Jarvis yesterday")
    assert not addressed("set a timer")
    assert addressed("Friday open notepad", "friday") and parse("Friday, open notepad", "Friday").args == {"name": "notepad"}


def test_replies():
    from jarvis.offline import Command

    assert reply(Command("x"), {"error": "boom"}) == "Sorry - boom"
    assert reply(Command("list_notes"), {"items": [{"text": "milk", "type": "todo"}]}) == "Your list: milk."
    assert reply(Command("list_notes"), {"result": "Nothing here yet."}) == "Nothing here yet."
    assert reply(Command("set_timer"), {"result": "Timer #1 started for 5 min."}) == "Timer #1 started for 5 min."


@pytest.fixture
def live(isolated_home, sandbox):
    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.live import LiveAssistant

    app = create_app(Settings(api_key="k", sound_effects=False, allowed_roots=[str(sandbox)]))
    a = LiveAssistant(app)
    a.spoken = []
    a.offline.say = a.spoken.append
    a.offline.listen = lambda: a.spoken.append("<listening>")
    a.events_seen = []
    app.events.subscribe(lambda k, d: a.events_seen.append((k, d)))
    yield a
    a.offline.close()


def wait_for(cond, secs=3.0):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_offline_commands_run_locally(live):
    live._failures, live._net_error = 1, True  # the last attempt failed: no network
    assert live.is_offline()
    live.send_text("milk list me daal do")
    assert wait_for(lambda: any(k == "assistant" for k, _ in live.events_seen))
    said = [d["text"] for k, d in live.events_seen if k == "assistant"][0]
    assert said.startswith("📴") and "to-do list" in said
    assert live.app.state.get("notes")[0]["text"] == "milk"
    assert live._pending == [] and live.spoken and "to-do" in live.spoken[0]
    assert any(k == "tool" and d["name"] == "add_note" for k, d in live.events_seen)


def test_other_messages_wait_for_the_internet(live):
    live._failures, live._net_error = 1, True
    live.send_text("what's the weather in Pune?")
    assert live._pending == ["what's the weather in Pune?"] or live._pending[-1].endswith("Pune?")
    assert any(k == "log" and "Offline" in d["text"] for k, d in live.events_seen)


def test_online_does_not_use_the_offline_path(live):
    live.send_text("milk list me daal do")  # not offline: goes to the model as usual
    time.sleep(0.2)
    assert not live.app.state.get("notes") and live._pending


def test_reminders_are_spoken_offline_not_saved_for_later(live):
    sent = []
    live.send_system = sent.append
    live._failures, live._net_error = 3, True
    live.notify("Reminder: drink water")
    assert live.spoken == ["Reminder: drink water"] and sent == []
    live._failures = 0
    live.notify("Reminder: stretch")
    assert sent and "stretch" in sent[0]


def test_listening_starts_after_two_failures_and_stops_when_back(live):
    live._on_connection_error(RuntimeError("no internet"), 1)
    assert "<listening>" not in live.spoken
    live._on_connection_error(RuntimeError("no internet"), 2)
    assert "<listening>" in live.spoken
    assert any(k == "offline_mode" and d["on"] for k, d in live.events_seen)
    live._offline_listening(False)
    assert any(k == "offline_mode" and not d["on"] for k, d in live.events_seen)


def test_practice_mode_stays_separate(live):
    live._failures, live._net_error = 1, True
    live.mode = "practice"
    assert live._run_offline("5 minute ka timer") is False


def test_a_rejected_setting_is_not_offline(live):
    live._on_connection_error(RuntimeError("received 1007 (invalid frame payload data) Request contains an "
                                           "invalid argument."), 1)
    live._on_connection_error(RuntimeError("1007 invalid argument"), 2)
    assert not live.is_offline() and "<listening>" not in live.spoken  # the internet is fine
    live._on_connection_error(OSError("getaddrinfo failed"), 4)
    assert live.is_offline()
