"""Phone control through Telegram (a fake Telegram; no network)."""

import threading
import time

import pytest

from jarvis.phone import PhoneBridge, PhoneError, TelegramAPI

ME = 555
STRANGER = 999
TOKEN = "123456789:AAH-secret-token-value-xyz"


class FakeAPI:
    def __init__(self):
        self.sent = []
        self.photos = []
        self.lock = threading.Lock()

    def call(self, method, params=None, files=None, timeout=15):
        with self.lock:
            if method == "getMe":
                return {"username": "my_jarvis_bot"}
            if method == "sendMessage":
                self.sent.append((params["chat_id"], params["text"], params.get("reply_markup")))
                return {"message_id": len(self.sent)}
            if method == "sendPhoto":
                self.photos.append(params["chat_id"])
            if method == "getUpdates":
                time.sleep(0.05)
                return []
            return {}

    def download(self, file_id):
        return b"OggS-fake-audio"

    def texts(self, chat=ME):
        return [t for c, t, _ in self.sent if c == chat]


class FakeChat:
    def __init__(self):
        self.got = []

    def send(self, text):
        self.got.append(text)
        return f"Done: {text.split(') ', 1)[-1]}"


@pytest.fixture
def bridge(isolated_home, sandbox):
    from jarvis.app import create_app
    from jarvis.config import Settings

    app = create_app(Settings(allowed_roots=[str(sandbox)], telegram_token=TOKEN))
    api = FakeAPI()
    chat = FakeChat()
    clock = {"now": 1_000_000.0}
    b = PhoneBridge(app, api=api, chat_factory=lambda app, ctx: chat, now=lambda: clock["now"])
    b.chat, b.clock = chat, clock
    yield b
    b.stop()


def msg(chat_id, text="", date=1_000_000, **extra):
    m = {"chat": {"id": chat_id, "type": "private"}, "from": {"id": chat_id, "first_name": "Musaddiq"},
         "date": date, "text": text}
    m.update(extra)
    return {"update_id": 1, "message": m}


def run_jobs(b):
    while not b._jobs.empty():
        kind, chat_id, payload = b._jobs.get()
        b._do(kind, chat_id, payload)


def test_pairing_with_the_code(bridge):
    b = bridge
    assert not b.paired
    code = b.begin_pairing()
    assert len(code) == 6 and code.isdigit()
    b.handle(msg(STRANGER, "000000" if code != "000000" else "111111"))  # wrong code
    assert not b.paired
    b.handle(msg(ME, f"/pair {code}"))
    assert b.paired and b.s.telegram_chat_id == ME and b.s.telegram_user == "Musaddiq"
    assert "Connected" in b.api.texts(ME)[-1]
    from jarvis.config import load_settings

    assert load_settings().telegram_chat_id == ME  # remembered


def test_code_expires(bridge):
    b = bridge
    code = b.begin_pairing()
    b.clock["now"] += 11 * 60
    b.handle(msg(ME, code))
    assert not b.paired


def test_strangers_are_refused_once(bridge):
    b = bridge
    b.s.telegram_chat_id = ME
    b.handle(msg(STRANGER, "shutdown the pc"))
    b.handle(msg(STRANGER, "please"))
    assert len(b.api.texts(STRANGER)) == 1 and "private assistant" in b.api.texts(STRANGER)[0]
    assert b._jobs.empty() and b.chat.got == []
    group = msg(ME, "hi")
    group["message"]["chat"]["type"] = "group"
    b.handle(group)  # never from a group, even with my id
    assert b._jobs.empty()


def test_commands_run_and_show_on_the_pc(bridge):
    b = bridge
    b.s.telegram_chat_id = ME
    seen = []
    b.app.events.subscribe(lambda k, d: seen.append((k, d)) if k == "phone" else None)
    b.handle(msg(ME, "set a timer for 10 minutes"))
    run_jobs(b)
    assert b.chat.got == ["(sent from the user's phone) set a timer for 10 minutes"]
    assert b.api.texts(ME)[-1] == "Done: set a timer for 10 minutes"
    assert [d["who"] for _, d in seen] == ["user", "assistant"]


def test_old_messages_are_not_run(bridge):
    b = bridge
    b.s.telegram_chat_id = ME
    b.handle(msg(ME, "shutdown the pc", date=1_000_000 - 3 * 3600))
    assert b._jobs.empty() and "skipped" in b.api.texts(ME)[-1]


def test_secrets_typed_on_the_phone_are_hidden(bridge):
    b = bridge
    b.s.telegram_chat_id = ME
    b.handle(msg(ME, "my card is 4111 1111 1111 1111 remember it"))
    run_jobs(b)
    assert "4111 1111 1111 1111" not in b.chat.got[0]


def test_voice_notes(bridge, monkeypatch):
    b = bridge
    b.s.telegram_chat_id = ME

    class LLM:
        def transcribe(self, audio, mime):
            assert audio.startswith(b"OggS") and mime == "audio/ogg"
            return "turn off the lights"

    b.app.ctx._llm = LLM()
    b.handle(msg(ME, "", voice={"file_id": "v1", "file_size": 2000}))
    run_jobs(b)
    assert "“turn off the lights”" in b.api.texts(ME)[0] and b.chat.got[-1].endswith("turn off the lights")
    b.handle(msg(ME, "", voice={"file_id": "v2", "file_size": 50 * 1024 * 1024}))
    assert "too long" in b.api.texts(ME)[-1]


def test_yes_no_buttons(bridge):
    b = bridge
    b.s.telegram_chat_id = ME
    answer = {}
    t = threading.Thread(target=lambda: answer.setdefault("v", b.ask_yes_no("Shut down the PC?", 3)))
    t.start()
    end = time.time() + 2
    while not b._answers and time.time() < end:
        time.sleep(0.01)
    qid = next(iter(b._answers))
    _, text, markup = b.api.sent[-1]
    assert "Shut down" in text and markup["inline_keyboard"][0][0]["callback_data"] == f"y:{qid}"
    b.handle({"update_id": 2, "callback_query": {"id": "c1", "data": f"y:{qid}",  # a stranger can't answer
                                                  "message": {"chat": {"id": STRANGER}, "message_id": 1}}})
    assert not answer
    b.handle({"update_id": 3, "callback_query": {"id": "c2", "data": f"y:{qid}",
                                                  "message": {"chat": {"id": ME}, "message_id": 1, "text": text}}})
    t.join(2)
    assert answer["v"] is True
    assert b.ask_yes_no("Delete everything?", 0.2) is False  # no answer = no


def test_risky_tools_ask_on_the_phone(bridge):
    b = bridge
    b.s.telegram_chat_id = ME
    ctx, images = b._context(ME)
    assert ctx.confirm is not b.app.ctx.confirm and ctx.attach_image is not b.app.ctx.attach_image
    ctx.attach_image(b"jpeg")
    assert images == [b"jpeg"]


def test_rate_limit_status_and_unpair(bridge):
    b = bridge
    b.s.telegram_chat_id = ME
    for i in range(25):
        b.handle(msg(ME, f"hello {i}"))
    assert "a lot at once" in b.api.texts(ME)[-1]
    while not b._jobs.empty():
        b._jobs.get()
    b.clock["now"] += 120
    b.app.state.set("reminders", [{"id": 1, "due": b.clock["now"] + 600, "message": "Call mom", "repeat": ""}])
    b.handle(msg(ME, "/status", date=b.clock["now"]))
    run_jobs(b)
    assert "running on your PC" in b.api.texts(ME)[-1] and "Call mom" in b.api.texts(ME)[-1]
    b.handle(msg(ME, "/unpair", date=b.clock["now"]))
    assert not b.paired and "Disconnected" in b.api.texts(ME)[-1]


def test_token_never_in_errors():
    class BoomHTTP:
        def post(self, url, **kw):
            raise ConnectionError(f"failed to connect to {url}")

        def get(self, url, **kw):
            raise ConnectionError(url)

    api = TelegramAPI(TOKEN, http=BoomHTTP())
    with pytest.raises(PhoneError) as err:
        api.call("getMe")
    assert TOKEN not in str(err.value) and "secret" not in str(err.value)

    class Resp:
        def __init__(self, code, body):
            self.status_code, self._b = code, body

        def json(self):
            return self._b

    class HTTP:
        def __init__(self, resp):
            self.resp = resp

        def post(self, url, **kw):
            return self.resp

    with pytest.raises(PhoneError, match="token is wrong"):
        TelegramAPI(TOKEN, http=HTTP(Resp(401, {"ok": False, "error_code": 401}))).call("getMe")
    with pytest.raises(PhoneError, match="Another program"):
        TelegramAPI(TOKEN, http=HTTP(Resp(409, {"ok": False, "error_code": 409}))).call("getUpdates")


def test_token_encrypted_and_not_backed_up(isolated_home):
    from jarvis import backup, secure
    from jarvis.config import Settings, load_settings, save_settings, settings_path

    save_settings(Settings(telegram_token=TOKEN, telegram_chat_id=ME))
    if secure.available():
        assert "secret-token" not in settings_path().read_text(encoding="utf-8")
    assert load_settings().telegram_token == TOKEN
    assert {"telegram_token", "telegram_chat_id"} <= set(backup.SECRET_SETTINGS)


def test_text_chat_trims_at_a_user_message():
    from google.genai import types

    from jarvis.text_chat import TextChat

    chat = TextChat.__new__(TextChat)
    user = lambda t: types.Content(role="user", parts=[types.Part(text=t)])  # noqa: E731
    call = types.Content(role="model", parts=[types.Part.from_function_call(name="x", args={})])
    reply = types.Content(role="user", parts=[types.Part.from_function_response(name="x", response={})])
    chat.history = [user("a"), call, reply] * 20
    chat._trim(keep=40)
    assert chat.history[0].parts[0].text == "a" and len(chat.history) <= 40
