"""Real-time voice conversation over the Gemini Live API.

The assistant runs its own asyncio loop (in a background thread when a window
is shown). Front ends talk to it only through these thread-safe methods:
send_text, set_muted, sleep, wake, restart, stop - and listen via app.events.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import sounddevice as sd
from google import genai
from google.genai import types

from jarvis.app import Jarvis
from jarvis.audio import MIC_BLOCK, MIC_RATE, Speaker, find_device, level
from jarvis.background import BackgroundServices
from jarvis.guardrails import describe, redact
from jarvis.sfx import SoundFX
from jarvis.voicecmd import is_sleep_command, mentions_name

SILENCE = bytes(MIC_BLOCK * 2)
log = logging.getLogger("jarvis.live")
ECHO_GUARD_SECONDS = 0.4  # keep the mic muted briefly after Jarvis stops talking
ANNOUNCE_SECONDS = 25.0  # how long an announcement may be spoken while asleep
GOODBYE_SECONDS = 2.5  # let "good night" be heard before going to sleep
NAME_HELLO_SECONDS = 4.0  # woken by name but no answer/action by then -> say "yes?"

# Errors that mean the resume handle is dead: reconnect with a fresh session instead.
STALE_SESSION_MARKERS = ("handle", "session not found", "expired", "1008", "invalid argument")

ASLEEP_NOTE = ("(system: asleep) You are now asleep. Stay completely silent and say nothing "
               "until you hear your name or get a '(system: awake)' note.")
AWAKE_HELLO = ("(system: awake) The user just woke you up. Say one very short line like "
               "'Yes?' or 'Haan, boliye?' in their language, then listen.")


def _a(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


class LiveAssistant:
    def __init__(self, app: Jarvis):
        self.app = app
        self.s = app.settings
        self.events = app.events
        self.client = genai.Client(api_key=self.s.effective_api_key)
        self.resume_handle: str | None = None
        self.mode = "agent"  # "agent" (Jarvis with commands) or "practice" (language tutor, no commands)
        self._handles: dict[str, str | None] = {}  # each mode resumes its own conversation
        self.practice = None  # jarvis.english.tutor.Tutor, created on first use
        self.awake = True
        self.muted = False
        self.last_activity = time.monotonic()
        self.last_spoke = 0.0
        self.state = ""
        self.wake = None
        self.session = None
        self.speaker: Speaker | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._session_task: asyncio.Task | None = None
        self._stopping = False
        self._restart_fresh = False
        self._connected_once = False
        self._reload_wake = bool(self.s.wake_word)  # loaded on the assistant thread, not the UI
        self._wake_name = ""
        self._failures = 0  # connection errors in a row
        # turned off if the model doesn't support expressive dialog (remembered per model)
        self._affective_ok = app.state.get("affective_unsupported") != self.s.live_model
        self._pending: list[str] = []  # texts typed before the session was up
        # texts sent before the server's first answer: if the server then rejects the session,
        # they are sent again after reconnecting instead of being lost
        self._unconfirmed: list[str] = []
        self._confirmed = False
        self._announced = False  # "Reconnected" shown for the current outage
        self._announce_until = 0.0  # reminders, timers, briefing and alerts are heard even while asleep
        self._tasks: set[asyncio.Task] = set()  # keep tool tasks alive until they finish
        self._session_mode = "agent"  # the mode the current session was opened in
        self._woken_by_name = False  # woke mid-speech: make sure Jarvis answers
        self._spoke_since_wake = False
        self.sfx = SoundFX(self.s.sound_effects, self.s.sound_volume,
                           find_device(self.s.output_device, "output")).attach(self.events)
        self.sfx.power_up = self.s.power_up_wake
        app.ctx.notify = self.notify
        app.ctx.attach_image = self._attach_image
        app.ctx.assistant = self
        self.background = BackgroundServices(app, self.announce)
        from jarvis.offline import OfflineVoice

        self.offline = OfflineVoice(on_heard=self._offline_heard, name=self.s.assistant_name)
        self._offline_note = False  # "offline - these still work" shown for the current outage

    # -- thread-safe controls -------------------------------------------------
    def _call(self, coro_fn, *args) -> None:
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(coro_fn(*args)))
        elif coro_fn == self._send_text and args:
            self._pending = (self._pending + [args[0]])[-10:]  # not started yet: sent once connected

    def send_text(self, text: str) -> None:
        """Typed input from the user. Critical secrets are masked before anything sees them."""
        text, hidden = redact(text)
        self.events.emit("user", text=text)
        if hidden:
            what = describe(hidden)
            self.events.emit("guardrail", action="hidden", tool="", text=what)
            text += (f" (system: the user typed {_a(what)} {what}; it was hidden for their safety. Don't ask for it - "
                     "remind them kindly never to share it, then help with the rest.)")
        if self.is_offline() and self._run_offline(text):
            return  # done on this computer, no internet needed
        if self.is_offline() and not self._offline_note:
            self._offline_note = True
            self.events.emit("log", text="📴 Offline - I'll answer this when the internet is back. Timers, "
                                          "reminders, notes, your list, volume and opening apps still work now.")
        was_asleep = not self.awake
        self._set_awake(True)
        if was_asleep:
            text = f"(system: awake - the user typed this) {text}"
        self._call(self._send_text, text)

    # -- offline ---------------------------------------------------------------------------
    def is_offline(self) -> bool:
        """No live session and the last attempt(s) failed."""
        return self.session is None and self._failures > 0

    def _run_offline(self, text: str) -> bool:
        """Run a simple command locally. True when it was one we understand."""
        if self.mode != "agent":
            return False
        from jarvis.offline import parse, reply

        cmd = parse(text, self.s.assistant_name)
        if cmd is None or cmd.tool not in self.registry.tools:
            return False

        def work():
            self.events.emit("tool_start", name=cmd.tool, args=cmd.args)
            result = self.registry.call(cmd.tool, cmd.args, self.app.ctx)
            self.events.emit("tool", name=cmd.tool, args=cmd.args, result=result)
            answer = reply(cmd, result)
            self.events.emit("assistant", text=f"📴 {answer}")
            if self.s.offline_voice and not self.muted:
                self.offline.say(answer)

        threading.Thread(target=work, daemon=True, name="jarvis-offline-cmd").start()
        return True

    def _offline_heard(self, said: str) -> None:
        """A voice command heard by the Windows recognizer while offline."""
        if self.is_offline() and not self.muted:
            self.send_text(said)

    def _offline_listening(self, on: bool) -> None:
        if on and self.s.offline_voice and self.mode == "agent" and not self.muted:
            self.offline.listen()
            self.events.emit("offline_mode", on=True)
        else:
            self.offline.stop_listening()
            if not on:
                self.events.emit("offline_mode", on=False)

    def send_system(self, text: str) -> None:
        """An instruction from the app itself (briefing, reminder, alert). Doesn't wake the mic."""
        self._call(self._send_text, text)

    def announce(self, text: str) -> None:
        """An app message the user must hear (briefing, alert, reminder) - even if Jarvis is asleep."""
        self._announce_until = time.monotonic() + ANNOUNCE_SECONDS
        self.send_system(text)

    def notify(self, message: str) -> None:
        self.events.emit("log", text=f"🔔 {message}")
        if self.is_offline():  # no model to say it: the Windows voice does (and it isn't told late again)
            if self.s.offline_voice:
                self.offline.say(message)
            return
        self.announce(f"(system notice) Tell the user now: {message}")

    def _audible(self) -> bool:
        return self.awake or time.monotonic() < self._announce_until

    def set_muted(self, muted: bool) -> None:
        self.muted = muted
        self._emit_state()

    def sleep(self, reason: str = "button") -> None:
        """Jarvis goes to sleep (never the computer)."""
        if not self.awake:
            return
        if reason in {"button", "idle"} and self.speaker:
            self.speaker.flush()
        self._set_awake(False)
        self.send_system(ASLEEP_NOTE)
        self.events.emit("log", text=f"💤 Sleeping - {self.how_to_wake()}.")

    def wake_up(self, reason: str = "button") -> None:
        if self.awake:
            return
        self._set_awake(True)
        if reason == "name":
            self._woken_by_name = True  # the model heard it; it answers on its own
            self._spoke_since_wake = False
            if self.loop:
                self.loop.call_later(NAME_HELLO_SECONDS, self._hello_if_quiet)
        else:
            self.send_system(AWAKE_HELLO)

    def _hello_if_quiet(self) -> None:
        """Only the name was said: answer with a short "yes?" instead of silence."""
        if self._woken_by_name and self.awake and not self._spoke_since_wake:
            self.send_system(AWAKE_HELLO)
        self._woken_by_name = False

    @property
    def wake_mode(self) -> str:
        """"wake_word" (offline model), "name" (say my name) or "" (button / typing only)."""
        if self.wake:
            return "wake_word"
        return "name" if self.s.name_wake else ""

    def how_to_wake(self) -> str:
        if self.wake_mode == "wake_word":
            return f'say "{self._wake_name.replace("_", " ")}" or press Wake'
        if self.wake_mode == "name":
            return f'say "{self.s.assistant_name}" or press Wake'
        return "press Wake or type a message"

    def set_mode(self, mode: str) -> None:
        """Switch between Jarvis (agent) and the language tutor (practice). Reconnects with the new role."""
        mode = "practice" if mode == "practice" else "agent"
        if mode == self.mode:
            return
        if mode == "practice" and self.practice is None:
            from jarvis.english.tutor import Tutor

            self.practice = Tutor(self.app)
        self._handles[self.mode] = self.resume_handle
        self.mode = mode
        self.resume_handle = self._handles.get(mode)
        self.events.emit("mode", mode=mode)
        self.restart()

    @property
    def registry(self):
        """The tools the model may use right now: the tutor's in practice mode, Jarvis's otherwise."""
        return self.practice.registry if self.mode == "practice" and self.practice else self.app.registry

    def restart(self, fresh: bool = False) -> None:
        """Reconnect with current settings (voice, devices). fresh=True forgets the conversation."""
        self._restart_fresh = fresh
        loaded = self._wake_name if self.wake else ""
        self._reload_wake = (self.s.wake_word or "") != loaded  # only when the wake word changed
        if self.loop and self._session_task:
            self.loop.call_soon_threadsafe(self._session_task.cancel)

    def stop(self) -> None:
        self._stopping = True
        self.offline.close()
        if self.loop and self._session_task:
            self.loop.call_soon_threadsafe(self._session_task.cancel)

    # -- helpers ----------------------------------------------------------------
    def _load_wake_word(self) -> None:
        self._reload_wake = False
        self.wake = None
        if not self.s.wake_word:
            return
        self.events.emit("log", text=f"Loading wake word '{self.s.wake_word}'...")
        try:
            from jarvis.wake import WakeWord

            self.wake = WakeWord(self.s.wake_word)
            self._wake_name = self.s.wake_word
            self.awake = False  # start asleep, waiting for the wake word
            self.events.emit("log", text=f"Say \"{self.s.wake_word.replace('_', ' ')}\" to wake me (or press Wake).")
        except Exception as exc:
            self.wake = None
            self.events.emit("log", text=f"[wake word off] {exc}")

    async def _send_text(self, text: str) -> None:
        if self.session:
            if not self._confirmed:
                self._unconfirmed = (self._unconfirmed + [text])[-10:]
            await self.session.send_realtime_input(text=text)
        else:
            self._pending.append(text)  # not connected yet: send as soon as we are
            self._pending = self._pending[-10:]

    def _attach_image(self, jpeg: bytes) -> None:
        """Called from a tool thread: send the image before the tool reply goes out."""
        if not (self.session and self.loop):
            raise RuntimeError("not connected")
        blob = types.Blob(data=jpeg, mime_type="image/jpeg")
        fut = asyncio.run_coroutine_threadsafe(self.session.send_realtime_input(video=blob), self.loop)
        fut.result(timeout=15)

    def _set_awake(self, awake: bool) -> None:
        self.awake = awake
        self.last_activity = time.monotonic()
        if not awake:
            self._woken_by_name = False
        self._emit_state()

    def _echo_guard(self, speaker: Speaker) -> bool:
        """True while Jarvis (or a sound effect) is audible, so the mic must not hear it."""
        if self.s.allow_interrupt:
            return False
        now = time.monotonic()
        return speaker.speaking or now - self.last_spoke < ECHO_GUARD_SECONDS or now < self.sfx.busy_until

    def _emit_state(self) -> None:
        if self.session is None:
            state = "connecting"
        elif not self.awake:
            state = "sleeping"
        elif self.speaker and self.speaker.speaking:
            state = "speaking"
        elif self.muted:
            state = "muted"
        else:
            state = "listening"
        if state != self.state:
            log.info("state %s -> %s", self.state or "-", state)
            self.state = state
            self.events.emit("state", state=state)

    def _config(self) -> types.LiveConnectConfig:
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=self.practice.prompt() if self.mode == "practice" and self.practice else self.app.prompt(),
            tools=[types.Tool(function_declarations=self.registry.declarations())],
            enable_affective_dialog=True if (self.s.expressive_voice and self._affective_ok) else None,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.s.voice)
                )
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            session_resumption=types.SessionResumptionConfig(handle=self.resume_handle),
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()
            ),
        )

    # -- tasks -------------------------------------------------------------------
    async def _send_mic(self, session) -> None:
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)

        def callback(indata, frames, t, status):
            data = bytes(indata)
            self.loop.call_soon_threadsafe(lambda: q.full() or q.put_nowait(data))

        device = find_device(self.s.input_device, "input")
        speaker = self.speaker
        with sd.RawInputStream(samplerate=MIC_RATE, channels=1, dtype="int16",
                               blocksize=MIC_BLOCK, device=device, callback=callback):
            while True:
                chunk = await q.get()
                self._emit_state()
                if speaker.speaking:
                    self.last_spoke = time.monotonic()
                mode = self.wake_mode
                if not self.awake:
                    if mode == "wake_word":
                        if self.wake.heard(chunk):
                            self.events.emit("log", text="[wake word heard]")
                            self.wake_up("wake_word")
                        continue  # offline: nothing leaves the computer while asleep
                    if mode != "name":
                        continue
                    # name mode: keep streaming so the transcript can catch the name
                elif mode and not speaker.speaking and \
                        time.monotonic() - self.last_activity > self.s.sleep_after_seconds:
                    self.sleep("idle")
                    continue
                silent = self.muted or self._echo_guard(speaker)
                if not silent and self.awake:
                    self.events.emit("level", source="mic", value=level(chunk))
                await session.send_realtime_input(
                    audio=types.Blob(data=SILENCE if silent else chunk, mime_type=f"audio/pcm;rate={MIC_RATE}")
                )

    async def _handle_tools(self, session, call: types.LiveServerToolCall) -> None:
        responses = []
        for fc in call.function_calls or []:
            args = dict(fc.args or {})
            self.events.emit("state", state="thinking")
            self.events.emit("tool_start", name=fc.name, args=args)
            registry = self.registry
            if fc.name not in registry.tools:  # e.g. an agent command while in practice mode
                result = {"error": "Practice mode is on, so computer commands are off. The user can press "
                                   "'Back to Jarvis' to use them again."}
            else:
                result = await asyncio.to_thread(registry.call, fc.name, args, self.app.ctx)
            self.events.emit("tool", name=fc.name, args=args, result=result)
            responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=result))
        self.state = ""
        await session.send_tool_response(function_responses=responses)

    def _user_turn(self, heard: list[str]) -> None:
        """A finished bit of user speech: show it and act on "go to sleep"."""
        text = "".join(heard).strip()
        if not text or not self.awake:
            return
        text, hidden = redact(text)
        self.events.emit("user", text=text)
        if hidden:  # said out loud: the model heard it, so make sure it won't repeat or use it
            what = describe(hidden)
            self.events.emit("guardrail", action="hidden", tool="", text=what)
            self.send_system(f"(system: the user just said {_a(what)} {what} out loud. Never repeat it or use it; "
                             "gently remind them to keep it private.)")
        if is_sleep_command(text):
            self.loop.call_later(GOODBYE_SECONDS, lambda: self.sleep("voice"))

    async def _receive(self, session) -> None:
        heard, said = [], []
        while True:
            async for msg in session.receive():
                if self._failures:  # back online: the Windows listener steps aside
                    self._offline_note = False
                    self._offline_listening(False)
                if self._failures and not self._announced:  # the server is talking to us again
                    self._announced = True
                    self.events.emit("log", text="✓ Reconnected.")
                    log.info("reconnected after %d failed attempt(s)", self._failures)
                if msg.server_content or msg.tool_call:
                    # the model really answered (setup messages alone don't count: a session can
                    # still be rejected after them) - what we sent has arrived
                    self._confirmed = True
                    self._unconfirmed = []
                    self._failures = 0
                    self._announced = False
                if (msg.session_resumption_update and msg.session_resumption_update.new_handle
                        and self.mode == self._session_mode):  # never store it under the other mode
                    self.resume_handle = msg.session_resumption_update.new_handle
                if msg.go_away:
                    return  # server will close soon; reconnect with the handle
                if msg.tool_call:
                    self._spoke_since_wake = True  # acting counts as answering
                    self.last_activity = time.monotonic()
                    task = asyncio.create_task(self._handle_tools(session, msg.tool_call))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                sc = msg.server_content
                if not sc:
                    continue
                if sc.interrupted:
                    self.speaker.flush()
                if sc.input_transcription and sc.input_transcription.text:
                    heard.append(sc.input_transcription.text)
                    if (not self.awake and self.wake_mode == "name"
                            and mentions_name("".join(heard), self.s.assistant_name)):
                        self.events.emit("log", text="[name heard]")
                        self.wake_up("name")
                    if self.awake:
                        self.last_activity = time.monotonic()
                        self.events.emit("caption", who="user", text=redact("".join(heard).strip())[0])
                if sc.output_transcription and sc.output_transcription.text and self._audible():
                    said.append(sc.output_transcription.text)
                    self.events.emit("caption", who="assistant", text=redact("".join(said).strip())[0])
                if sc.model_turn:
                    if heard:  # show what the user said as soon as the reply starts
                        self._user_turn(heard)
                        heard = []
                    for part in sc.model_turn.parts or []:
                        # while asleep the reply is dropped (Jarvis stays quiet) - except announcements
                        if part.inline_data and part.inline_data.data and self._audible():
                            self.speaker.play(part.inline_data.data)
                            self.last_activity = time.monotonic()
                            self._spoke_since_wake = True
                if sc.turn_complete:
                    self._user_turn(heard)
                    if said and self._audible():
                        self.events.emit("assistant", text=redact("".join(said).strip())[0])
                    self.events.emit("caption", who="", text="")
                    heard, said = [], []

    async def _session_once(self) -> None:
        if self.speaker is None:
            self.speaker = Speaker(find_device(self.s.output_device, "output"),
                                   on_level=lambda v: self.events.emit("level", source="speaker", value=v))
            self.speaker.start()
        self._session_mode = self.mode
        async with self.client.aio.live.connect(model=self.s.live_model, config=self._config()) as session:
            self.session = session
            self._confirmed = False
            self._unconfirmed = []
            pending, self._pending = self._pending, []
            for text in pending:
                await self._send_text(text)
            self._emit_state()
            if not self._connected_once:
                self._connected_once = True
                self.events.emit("log", text=f"Connected ({self.s.live_model}, voice {self.s.voice}).")
                self.background.on_connected()
            mic = asyncio.create_task(self._send_mic(session))
            recv = asyncio.create_task(self._receive(session))
            try:
                done, _ = await asyncio.wait({mic, recv}, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()  # re-raise errors
            finally:
                mic.cancel()
                recv.cancel()
                self.session = None

    def _on_connection_error(self, exc: Exception, backoff: float) -> None:
        self._failures += 1
        if self._failures == 2:  # really offline, not a blip: simple voice commands without internet
            self._offline_listening(True)
        if self._unconfirmed:  # never answered: send again after reconnecting
            self._pending = (self._unconfirmed + self._pending)[-10:]
            self._unconfirmed = []
        text = str(exc)
        low = text.lower()
        if self._affective_ok and self.s.expressive_voice and ("affective" in low or "invalid argument" in low):
            # this model can't do expressive dialog: connect without it (and remember, so we don't try again)
            self._affective_ok = False
            self.app.state.set("affective_unsupported", self.s.live_model)
            self.events.emit("log", text="(Expressive voice isn't supported by this model - using the normal voice.)")
        if "voice" in low and ("not" in low or "invalid" in low or "unsupported" in low) and self.s.voice != "Charon":
            self.events.emit("log", text=f"(Voice '{self.s.voice}' isn't available here - using Charon.)")
            self.s.voice = "Charon"
        log.warning("connection error #%d: %s: %s", self._failures, type(exc).__name__, text)
        if self.resume_handle and (self._failures >= 2 or any(m in text.lower() for m in STALE_SESSION_MARKERS)):
            # an expired session can never be resumed: start fresh rather than retrying forever
            self.resume_handle = None
            log.info("dropped the resume handle; next attempt starts a new session")
        self.events.emit("log", text=f"[connection lost] {type(exc).__name__}: {text[:120]} - retrying in {backoff}s")

    # -- main loop ----------------------------------------------------------------
    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.background.start()
        self._emit_state()
        backoff = 1
        try:
            while not self._stopping:
                if self._reload_wake:
                    await asyncio.to_thread(self._load_wake_word)
                    self._emit_state()
                self._session_task = asyncio.create_task(self._session_once())
                try:
                    await self._session_task
                    backoff = 1
                except asyncio.CancelledError:
                    if self._stopping:
                        break
                    # restart(): new voice/devices/prompt
                    if self.speaker:
                        self.speaker.close()
                        self.speaker = None
                    if self._restart_fresh:
                        self.resume_handle = None
                    self.events.emit("log", text="Reconnecting with new settings...")
                    continue
                except Exception as exc:
                    self._on_connection_error(exc, backoff)
                    self.session = None
                    self._emit_state()
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
        finally:
            self.session = None
            self.background.stop()
            if self.speaker:
                self.speaker.close()
            self.events.emit("state", state="offline")
            self.sfx.close()
            self.background.save_summary()  # directly: thread pools can't start during interpreter exit

    def run_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=lambda: asyncio.run(self.run()), name="jarvis-live", daemon=True)
        thread.start()
        return thread


def run_voice(app: Jarvis) -> None:
    """Terminal voice mode (no window)."""
    from jarvis.events import console_printer

    app.events.subscribe(console_printer(app.settings.assistant_name))
    assistant = LiveAssistant(app)
    print(f"{app.settings.assistant_name} is listening. Press Ctrl+C to quit.")
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        print("\nBye!")
