"""Say-the-name wake up, "so jao" sleep, sound effects and the live receive loop."""

import asyncio
from types import SimpleNamespace as NS

import pytest

from jarvis.voicecmd import is_sleep_command, mentions_name


@pytest.mark.parametrize("text", [
    "Jarvis, what time is it?", "hey jarvish", "जार्विस, टाइम बताओ", "ok JARVIS!",
])
def test_name_is_heard(text):
    assert mentions_name(text, "Jarvis")


@pytest.mark.parametrize("text", ["Charvis", "jarvisss open chrome", "", "service"])
def test_other_words_are_not_the_name(text):
    assert not mentions_name(text, "Jarvis")


def test_custom_name():
    assert mentions_name("Friday, open YouTube", "Friday")
    assert not mentions_name("Friday, open YouTube", "Jarvis")


@pytest.mark.parametrize("text", [
    "so jao", "Jarvis so jao", "go to sleep", "chup ho jao yaar", "सो जाओ।", "سو جاؤ", "stop listening",
])
def test_sleep_commands(text):
    assert is_sleep_command(text)


@pytest.mark.parametrize("text", [
    "pc ko so jao", "computer go to sleep", "put my laptop to sleep", "mujhe so jana hai",
    "I can't go to sleep because the movie I am watching is too good", "hello",
])
def test_not_sleep_commands(text):
    assert not is_sleep_command(text)


# -- sound effects ----------------------------------------------------------------

def test_sounds_are_built_and_short():
    from jarvis.sfx import SoundFX

    fx = SoundFX()
    for name in ("startup", "wake", "sleep", "action", "success", "error", "notify", "alert"):
        assert 0 < fx.duration(name) < 1.0
        assert abs(fx.sounds[name]).max() <= 0.61


def test_effects_follow_events(monkeypatch):
    from jarvis.sfx import SoundFX

    fx = SoundFX()
    played = []
    monkeypatch.setattr(fx, "play", played.append)
    for state in ("connecting", "listening", "speaking", "sleeping", "listening"):
        fx.on_event("state", {"state": state})
    fx.on_event("tool_start", {"name": "open_app"})
    fx.on_event("tool", {"name": "open_app", "result": {"result": "ok"}})
    fx.on_event("tool", {"name": "open_app", "result": {"error": "nope"}})
    fx.on_event("log", {"text": "🔔 Reminder: tea"})  # the reminder chime comes with reminder_due instead
    fx.on_event("reminder_due", {"id": 1, "message": "tea"})
    fx.on_event("log", {"text": "🔔 Timer done!"})
    assert played == ["startup", "sleep", "powerup", "action", "success", "error", "reminder", "notify"]


def test_disabled_effects_are_silent():
    from jarvis.sfx import SoundFX

    fx = SoundFX(enabled=False)
    fx.play("wake")
    assert fx._thread is None and fx.busy_until == 0


# -- live assistant ----------------------------------------------------------------

class FakeSpeaker:
    def __init__(self):
        self.chunks = []
        self.speaking = False

    def play(self, chunk):
        self.chunks.append(chunk)

    def flush(self):
        pass


class Stop(Exception):
    pass


class FakeSession:
    def __init__(self, messages):
        self.messages = messages
        self.used = False

    async def receive(self):
        if self.used:
            raise Stop
        self.used = True
        for m in self.messages:
            yield m


def msg(heard=None, audio=None, said=None, done=False):
    parts = [NS(inline_data=NS(data=audio))] if audio else None
    return NS(
        session_resumption_update=None, go_away=None, tool_call=None,
        server_content=NS(
            interrupted=False,
            input_transcription=NS(text=heard) if heard else None,
            output_transcription=NS(text=said) if said else None,
            model_turn=NS(parts=parts) if parts else None,
            turn_complete=done,
        ),
    )


@pytest.fixture
def live():
    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.live import LiveAssistant

    app = create_app(Settings(api_key="test-key", sound_effects=False))
    a = LiveAssistant(app)
    a.speaker = FakeSpeaker()
    a.sent = []
    a.send_system = a.sent.append
    events = []
    app.events.subscribe(lambda k, d: events.append((k, d)))
    a.events_seen = events
    return a


def run_receive(a, messages):
    async def go():
        a.loop = asyncio.get_running_loop()
        with pytest.raises(Stop):
            await a._receive(FakeSession(messages))
    asyncio.run(go())


def test_saying_the_name_wakes_and_answers(live):
    live.sleep()
    assert not live.awake and live.wake_mode == "name"
    assert live.sent[-1].startswith("(system: asleep)")
    run_receive(live, [msg(heard="Jarvis, "), msg(heard="time kya hai?"),
                       msg(audio=b"\x01\x02", said="Saade das baje hain."), msg(done=True)])
    assert live.awake
    assert live.speaker.chunks == [b"\x01\x02"]
    users = [d["text"] for k, d in live.events_seen if k == "user"]
    assert users == ["Jarvis, time kya hai?"]


def test_asleep_ignores_other_speech(live):
    live.sleep()
    run_receive(live, [msg(heard="aaj mausam kaisa hai"), msg(audio=b"\x09", said="Aaj dhoop hai."),
                       msg(done=True)])
    assert not live.awake
    assert live.speaker.chunks == []  # stays quiet
    assert not [k for k, _ in live.events_seen if k in {"user", "assistant"}]


def test_name_alone_gets_a_short_hello(live):
    from jarvis.live import AWAKE_HELLO

    live.sleep()
    later = []

    async def go():
        live.loop = asyncio.get_running_loop()
        live.loop.call_later = lambda delay, fn: later.append(fn)
        with pytest.raises(Stop):
            await live._receive(FakeSession([msg(heard="Jarvis"), msg(done=True)]))
    asyncio.run(go())
    assert live.awake and len(later) == 1
    later[0]()
    assert live.sent[-1] == AWAKE_HELLO


def test_no_extra_hello_when_jarvis_acts(live):
    from jarvis.live import AWAKE_HELLO

    live.sleep()
    live.loop = None
    live.wake_up("name")
    live._spoke_since_wake = True  # e.g. a tool call or speech came first
    live._hello_if_quiet()
    assert AWAKE_HELLO not in live.sent


def test_so_jao_puts_jarvis_to_sleep_not_the_pc(live, monkeypatch):
    import jarvis.tools.system as system

    monkeypatch.setattr(system.subprocess, "Popen", lambda *a: (_ for _ in ()).throw(AssertionError("power")))
    later = []

    async def go():
        live.loop = asyncio.get_running_loop()
        monkeypatch.setattr(live.loop, "call_later", lambda delay, fn: later.append(fn))
        with pytest.raises(Stop):
            await live._receive(FakeSession([msg(heard="Jarvis so jao"), msg(done=True)]))
    asyncio.run(go())
    assert live.awake and len(later) == 1  # goodbye first
    later[0]()
    assert not live.awake


def test_wake_modes(live):
    assert "Jarvis" in live.how_to_wake()
    live.s.name_wake = False
    assert live.wake_mode == "" and "press Wake" in live.how_to_wake()
    live.wake = object()
    live._wake_name = "hey_jarvis"
    assert live.wake_mode == "wake_word" and "hey jarvis" in live.how_to_wake()


def test_button_wake_says_hello_and_typing_wakes(live):
    from jarvis.live import AWAKE_HELLO

    live.sleep()
    live.wake_up()
    assert live.awake and live.sent[-1] == AWAKE_HELLO
    live.sleep()
    live.send_text("hi")
    assert live.awake


def test_sound_effects_count_for_echo_guard(live):
    import time

    assert not live._echo_guard(live.speaker)
    live.sfx.busy_until = time.monotonic() + 1
    assert live._echo_guard(live.speaker)
    live.s.allow_interrupt = True
    assert not live._echo_guard(live.speaker)


def test_text_typed_before_connecting_is_kept(live):
    asyncio.run(live._send_text("hello"))
    assert live._pending == ["hello"]


def test_expired_session_reconnects_fresh(live):
    live.resume_handle = "old-handle"
    live._on_connection_error(Exception("1008 None. BidiGenerateContent session not found or expired"), 1)
    assert live.resume_handle is None


def test_repeated_errors_drop_the_handle(live):
    live.resume_handle = "old-handle"
    live._on_connection_error(Exception("network down"), 1)
    assert live.resume_handle == "old-handle"  # one blip: try to resume the conversation
    live._on_connection_error(Exception("network down"), 2)
    assert live.resume_handle is None


def test_text_is_resent_when_the_server_rejects_the_session(live):
    class Session:
        def __init__(self):
            self.sent = []

        async def send_realtime_input(self, text=None, **kw):
            self.sent.append(text)

    live.session = Session()
    asyncio.run(live._send_text("what is 15 times 12?"))
    assert live._unconfirmed == ["what is 15 times 12?"]
    live._on_connection_error(Exception("1007 invalid argument"), 1)
    assert live._pending == ["what is 15 times 12?"] and live._unconfirmed == []


def test_answered_text_is_not_resent(live):
    live.session = type("S", (), {"send_realtime_input": lambda self, **k: asyncio.sleep(0)})()
    asyncio.run(live._send_text("hello"))
    run_receive(live, [msg(said="hi", done=True)])  # the server answered
    live._on_connection_error(Exception("network"), 1)
    assert live._pending == []


def test_reminders_are_heard_even_while_asleep(live):
    live.sleep()
    live.notify("Reminder: drink water")  # opens a short window in which Jarvis may speak
    run_receive(live, [msg(audio=b"\x05", said="Time to drink water!"), msg(done=True)])
    assert live.speaker.chunks == [b"\x05"]
    assert any(k == "assistant" for k, _ in live.events_seen)
    assert not live.awake  # still asleep afterwards


def test_ambient_replies_stay_silent_without_an_announcement(live):
    live.sleep()
    run_receive(live, [msg(audio=b"\x06", said="hmm"), msg(done=True)])
    assert live.speaker.chunks == []


def test_text_typed_before_the_loop_starts_is_kept(isolated_home):
    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.live import LiveAssistant

    a = LiveAssistant(create_app(Settings(api_key="k", sound_effects=False)))
    a.send_text("hello early")
    assert a._pending == ["hello early"]


def test_resume_handle_is_not_stored_under_the_wrong_mode(live):
    from types import SimpleNamespace as NS2

    live._session_mode = "agent"
    live.mode = "practice"  # switched while the old session still talks
    update = NS(session_resumption_update=NS2(new_handle="agent-h"), go_away=None, tool_call=None, server_content=None)
    run_receive(live, [update])
    assert live.resume_handle is None
