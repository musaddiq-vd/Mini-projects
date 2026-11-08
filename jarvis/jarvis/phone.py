"""Control Jarvis from your phone through your own Telegram bot.

Security:
- only the one Telegram account that paired with a 6-digit code (shown on the PC) is obeyed; others are refused
- risky actions (shutdown, deleting, sending messages) ask for a tap on Yes / No in the chat
- commands that arrived while the PC was off are not run late; Jarvis says it skipped them
- the bot token is encrypted on disk and never written to logs or error messages
- everything sent from the phone, and every reply, also shows in the Jarvis window

Text messages and voice notes run through the same tools as the voice assistant, in their own context, so the
phone and the live voice session never get in each other's way.
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import random
import threading
import time
from collections import deque
from collections.abc import Callable

log = logging.getLogger("jarvis.phone")
logging.getLogger("urllib3").setLevel(logging.WARNING)  # request URLs contain the bot token

API = "https://api.telegram.org"
STALE_S = 10 * 60  # older messages (sent while the PC was off) are not acted on
PAIR_S = 10 * 60
MAX_PER_MIN = 20
MAX_TEXT = 3900
CONFIRM_S = 90
HELP = ("Hi! I'm {name} on your PC. Send me what you'd say out loud - \"set a timer for 10 minutes\", \"what's "
        "my battery?\", \"turn off the bedroom light\" - or a voice note.\n\n/status - how the PC is doing\n"
        "/screen - a picture of the screen\n/unpair - disconnect this chat")


class PhoneError(Exception):
    """A friendly problem that never contains the bot token."""


class TelegramAPI:
    def __init__(self, token: str, http=None):
        self.token = token
        if http is None:
            import requests

            http = requests
        self.http = http

    def _url(self, method: str) -> str:
        return f"{API}/bot{self.token}/{method}"

    def call(self, method: str, params: dict | None = None, files: dict | None = None, timeout: float = 15):
        try:
            if files:
                resp = self.http.post(self._url(method), data=params or {}, files=files, timeout=timeout)
            else:
                resp = self.http.post(self._url(method), json=params or {}, timeout=timeout)
        except Exception as exc:
            name = type(exc).__name__
            raise PhoneError("Telegram didn't answer in time." if "Timeout" in name
                             else "Couldn't reach Telegram (no internet?).") from None
        try:
            data = resp.json()
        except Exception:
            data = {}
        code = getattr(resp, "status_code", 200)
        if code == 401 or data.get("error_code") == 401:
            raise PhoneError("Telegram says the bot token is wrong. Copy it again from @BotFather.")
        if code == 409 or data.get("error_code") == 409:
            raise PhoneError("Another program is already reading this bot. Close it (or use a different bot).")
        if code == 403 or data.get("error_code") == 403:
            raise PhoneError("The chat blocked the bot.")
        if not data.get("ok"):
            raise PhoneError(f"Telegram error: {str(data.get('description', code))[:120]}")
        return data.get("result")

    def download(self, file_id: str) -> bytes:
        info = self.call("getFile", {"file_id": file_id})
        path = (info or {}).get("file_path")
        if not path:
            raise PhoneError("Couldn't get that file from Telegram.")
        try:
            resp = self.http.get(f"{API}/file/bot{self.token}/{path}", timeout=30)
        except Exception:
            raise PhoneError("Couldn't download the voice note.") from None
        if getattr(resp, "status_code", 200) >= 400:
            raise PhoneError("Couldn't download the voice note.")
        return resp.content


class PhoneConfirm:
    """Asks Yes / No in the Telegram chat and waits for the tap (used instead of the window's banner)."""

    def __init__(self, bridge: PhoneBridge, timeout: float = CONFIRM_S):
        self.bridge = bridge
        self.timeout = timeout

    def confirm(self, question: str) -> bool:
        return self.bridge.ask_yes_no(question, self.timeout)


class PhoneBridge:
    def __init__(self, app, api: TelegramAPI | None = None, chat_factory: Callable | None = None,
                 now: Callable[[], float] = time.time):
        self.app = app
        self.s = app.settings
        self.api = api or TelegramAPI(self.s.telegram_token)
        self.now = now
        self._stop = threading.Event()
        self._jobs: queue.Queue = queue.Queue(maxsize=50)
        self._offset = 0
        self._pair_code = ""
        self._pair_until = 0.0
        self._answers: dict[str, threading.Event] = {}
        self._answer_value: dict[str, bool] = {}
        self._times: deque[float] = deque(maxlen=MAX_PER_MIN)
        self._warned: set[int] = set()
        self._chat = None
        self._chat_factory = chat_factory
        self.bot_name = ""
        self.status_text = ""
        self.threads: list[threading.Thread] = []

    # -- life cycle ---------------------------------------------------------------------------
    def start(self) -> None:
        for target in (self._poll_loop, self._work_loop):
            t = threading.Thread(target=target, daemon=True, name=f"jarvis-phone-{target.__name__}")
            t.start()
            self.threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass
        for ev in list(self._answers.values()):
            ev.set()

    def _emit(self, kind: str, **data) -> None:
        if self.app.events is not None:
            try:
                self.app.events.emit(kind, **data)
            except Exception:
                pass

    # -- pairing -------------------------------------------------------------------------------
    def check_token(self) -> str:
        """The bot's @username (raises PhoneError when the token is wrong)."""
        me = self.api.call("getMe") or {}
        self.bot_name = me.get("username", "")
        return self.bot_name

    def begin_pairing(self) -> str:
        self._pair_code = f"{random.SystemRandom().randint(0, 999999):06d}"
        self._pair_until = self.now() + PAIR_S
        return self._pair_code

    @property
    def paired(self) -> bool:
        return bool(self.s.telegram_chat_id)

    def _save_pair(self, chat_id: int, who: str) -> None:
        from jarvis.config import save_settings

        self.s.telegram_chat_id = int(chat_id)
        self.s.telegram_user = who
        self._pair_code = ""
        try:
            save_settings(self.s)
        except OSError:
            log.warning("couldn't save the pairing")
        self._emit("phone_paired", who=who)

    # -- receiving -----------------------------------------------------------------------------
    def _poll_loop(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            try:
                updates = self.api.call("getUpdates", {"offset": self._offset, "timeout": 25,
                                                       "allowed_updates": ["message", "callback_query"]},
                                        timeout=35) or []
                backoff = 2.0
                self.status_text = "connected"
            except PhoneError as exc:
                self.status_text = str(exc)
                self._emit("phone_status", text=str(exc))
                if self._stop.wait(backoff):
                    return
                backoff = min(60.0, backoff * 2)
                continue
            for u in updates:
                self._offset = max(self._offset, int(u.get("update_id", 0)) + 1)
                try:
                    self.handle(u)
                except Exception:  # one odd update never stops the bridge
                    log.exception("phone update failed")

    def handle(self, u: dict) -> None:
        if "callback_query" in u:
            self._on_button(u["callback_query"])
            return
        msg = u.get("message") or {}
        chat = msg.get("chat") or {}
        chat_id = chat.get("id")
        sender = msg.get("from") or {}
        if chat_id is None or sender.get("is_bot") or chat.get("type") != "private":
            return
        text = (msg.get("text") or "").strip()
        who = sender.get("first_name") or sender.get("username") or "you"
        if not self.paired or chat_id != self.s.telegram_chat_id:
            code = text.replace("/pair", "").replace("/start", "").strip()
            if (not self.paired and self._pair_code and self.now() < self._pair_until
                    and code == self._pair_code):
                self._save_pair(chat_id, who)
                self.send(chat_id, f"✅ Connected! I'll only listen to you, {who}.\n\n"
                          + HELP.format(name=self.s.assistant_name))
            elif chat_id not in self._warned:
                self._warned.add(chat_id)
                self.send(chat_id, "This is a private assistant. To connect, open Jarvis on the PC → Settings → "
                                   "Phone and send the code shown there.")
            return
        if self.now() - float(msg.get("date", self.now())) > STALE_S:
            when = time.strftime("%I:%M %p", time.localtime(float(msg.get("date", 0)))).lstrip("0")
            self.send(chat_id, f"⏸ I was offline and skipped your message from {when}. Send it again if you still "
                               "want it.")
            return
        if text in ("/start", "/help"):
            self.send(chat_id, HELP.format(name=self.s.assistant_name))
            return
        if text == "/unpair":
            self.send(chat_id, "👋 Disconnected. Pair again from Settings → Phone on the PC.")
            self._save_pair(0, "")
            return
        self._times.append(self.now())
        if len(self._times) == MAX_PER_MIN and self.now() - self._times[0] < 60:
            self.send(chat_id, "⏳ That's a lot at once - give me a minute.")
            return
        job = None
        if text == "/status":
            job = ("status", chat_id, "")
        elif text == "/screen":
            job = ("screen", chat_id, "")
        elif text:
            job = ("text", chat_id, text)
        elif msg.get("voice") or msg.get("audio"):
            media = msg.get("voice") or msg.get("audio")
            if int(media.get("file_size", 0) or 0) > 10 * 1024 * 1024:
                self.send(chat_id, "That voice note is too long - keep it under a few minutes.")
                return
            job = ("voice", chat_id, media.get("file_id", ""))
        else:
            self.send(chat_id, "I can read text messages and voice notes.")
            return
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            self.send(chat_id, "I'm busy with earlier messages - try again in a moment.")

    # -- doing ---------------------------------------------------------------------------------
    def _context(self, chat_id: int):
        """The tools' context for phone commands: same tools, phone confirmations, own images."""
        images: list[bytes] = []
        ctx = dataclasses.replace(self.app.ctx, confirm=PhoneConfirm(self),
                                  attach_image=images.append,
                                  notify=lambda m: self.send(chat_id, f"🔔 {m}"))
        return ctx, images

    def _engine(self, ctx):
        if self._chat_factory is not None:
            return self._chat_factory(self.app, ctx)
        if self._chat is None:
            from jarvis.text_chat import TextChat

            self._chat = TextChat(self.app, ctx=ctx)
        self._chat.ctx = ctx
        return self._chat

    def _work_loop(self) -> None:
        while not self._stop.is_set():
            job = self._jobs.get()
            if job is None:
                return
            kind, chat_id, payload = job
            try:
                self._do(kind, chat_id, payload)
            except PhoneError as exc:
                self.send(chat_id, f"⚠ {exc}")
            except Exception as exc:
                log.exception("phone job failed")
                self.send(chat_id, f"⚠ Something went wrong ({type(exc).__name__}).")

    def _do(self, kind: str, chat_id: int, payload: str) -> None:
        self._action(chat_id, "typing")
        if kind == "status":
            self.send(chat_id, self.status_report())
            return
        if kind == "screen":
            from jarvis.tools.vision import screenshot_jpeg

            jpeg = screenshot_jpeg()
            self._emit("capture", what="screen")
            self._emit("phone", who="user", text="/screen (sent a screenshot to your phone)")
            self.api.call("sendPhoto", {"chat_id": chat_id, "caption": "Your screen right now"},
                          files={"photo": ("screen.jpg", jpeg, "image/jpeg")}, timeout=60)
            return
        text = payload
        if kind == "voice":
            audio = self.api.download(payload)
            text = self.app.ctx.llm.transcribe(audio, "audio/ogg")
            if not text:
                self.send(chat_id, "I couldn't hear anything in that voice note.")
                return
            self.send(chat_id, f"🎙 “{text[:300]}”")
        from jarvis.guardrails import redact

        text, _ = redact(text)  # a secret typed on the phone is hidden before anything sees it
        self._emit("phone", who="user", text=text)
        ctx, images = self._context(chat_id)
        reply = self._engine(ctx).send(f"(sent from the user's phone) {text}") or "Done."
        self._emit("phone", who="assistant", text=reply)
        self.send(chat_id, reply)
        for jpeg in images[-1:]:  # what Jarvis looked at, so the phone sees it too
            try:
                self.api.call("sendPhoto", {"chat_id": chat_id}, files={"photo": ("seen.jpg", jpeg, "image/jpeg")},
                              timeout=60)
            except PhoneError:
                pass

    def status_report(self) -> str:
        import psutil

        lines = [f"🖥 {self.s.assistant_name} is running on your PC."]
        battery = psutil.sensors_battery()
        if battery:
            lines.append(f"🔋 Battery {battery.percent:.0f}%" + (" (charging)" if battery.power_plugged else ""))
        lines.append(f"🧠 CPU {psutil.cpu_percent(interval=0.3):.0f}% · RAM {psutil.virtual_memory().percent:.0f}%")
        st = self.app.state
        if st is not None:
            now = self.now()
            rem = sorted((r for r in st.get("reminders", []) or [] if r.get("due", 0) > now), key=lambda r: r["due"])
            if rem:
                when = time.strftime("%I:%M %p", time.localtime(rem[0]["due"])).lstrip("0")
                lines.append(f"⏰ Next reminder: {rem[0]['message']} at {when}")
            todos = [n for n in st.get("notes", []) or [] if n.get("todo") and not n.get("done")]
            if todos:
                lines.append(f"📝 {len(todos)} open to-do{'s' if len(todos) != 1 else ''}")
        return "\n".join(lines)

    # -- Yes / No buttons ----------------------------------------------------------------------
    def ask_yes_no(self, question: str, timeout: float) -> bool:
        chat_id = self.s.telegram_chat_id
        if not chat_id:
            return False
        qid = f"{random.SystemRandom().getrandbits(48):x}"
        ev = threading.Event()
        self._answers[qid] = ev
        markup = {"inline_keyboard": [[{"text": "✅ Yes", "callback_data": f"y:{qid}"},
                                       {"text": "❌ No", "callback_data": f"n:{qid}"}]]}
        try:
            self.api.call("sendMessage", {"chat_id": chat_id, "text": f"⚠ {question}", "reply_markup": markup})
        except PhoneError:
            self._answers.pop(qid, None)
            return False
        ev.wait(timeout)
        self._answers.pop(qid, None)
        return self._answer_value.pop(qid, False)

    def _on_button(self, q: dict) -> None:
        chat_id = ((q.get("message") or {}).get("chat") or {}).get("id")
        data = q.get("data", "")
        try:
            self.api.call("answerCallbackQuery", {"callback_query_id": q.get("id")})
        except PhoneError:
            pass
        if chat_id != self.s.telegram_chat_id or ":" not in data:
            return
        yes_no, qid = data.split(":", 1)
        ev = self._answers.get(qid)
        if ev is None:
            return  # an old button: the question already timed out
        self._answer_value[qid] = yes_no == "y"
        ev.set()
        msg = q.get("message") or {}
        try:
            self.api.call("editMessageText", {"chat_id": chat_id, "message_id": msg.get("message_id"),
                                              "text": f"{msg.get('text', '')}\n\n{'✅ Yes' if yes_no == 'y' else '❌ No'}"})
        except PhoneError:
            pass

    # -- sending -------------------------------------------------------------------------------
    def _action(self, chat_id: int, action: str) -> None:
        try:
            self.api.call("sendChatAction", {"chat_id": chat_id, "action": action})
        except PhoneError:
            pass

    def send(self, chat_id: int, text: str) -> None:
        text = text or "…"
        for i in range(0, min(len(text), MAX_TEXT * 3), MAX_TEXT):
            try:
                self.api.call("sendMessage", {"chat_id": chat_id, "text": text[i:i + MAX_TEXT]})
            except PhoneError as exc:
                log.warning("phone send failed: %s", exc)
                return
