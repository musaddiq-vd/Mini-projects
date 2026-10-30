"""Real-time voice conversation over the Gemini Live API.

The assistant runs its own asyncio loop (in a background thread when a window
is shown). Front ends talk to it only through these thread-safe methods:
send_text, set_muted, sleep, wake, restart, stop - and listen via app.events.
"""

from __future__ import annotations

import asyncio
import threading
import time

import sounddevice as sd
from google import genai
from google.genai import types

from jarvis.app import Jarvis
from jarvis.audio import MIC_BLOCK, MIC_RATE, Speaker, find_device, level
from jarvis.background import BackgroundServices

SILENCE = bytes(MIC_BLOCK * 2)
ECHO_GUARD_SECONDS = 0.4  # keep the mic muted briefly after Jarvis stops talking


class LiveAssistant:
    def __init__(self, app: Jarvis):
        self.app = app
        self.s = app.settings
        self.events = app.events
        self.client = genai.Client(api_key=self.s.effective_api_key)
        self.resume_handle: str | None = None
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
        app.ctx.notify = self.notify
        app.ctx.attach_image = self._attach_image
        app.ctx.assistant = self
        self.background = BackgroundServices(app, self.send_system)

    # -- thread-safe controls -------------------------------------------------
    def _call(self, coro_fn, *args) -> None:
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(coro_fn(*args)))

    def send_text(self, text: str) -> None:
        """Typed input from the user."""
        self.events.emit("user", text=text)
        self._set_awake(True)
        self._call(self._send_text, text)

    def send_system(self, text: str) -> None:
        """An instruction from the app itself (briefing, reminder, alert). Doesn't wake the mic."""
        self._call(self._send_text, text)

    def notify(self, message: str) -> None:
        self.events.emit("log", text=f"🔔 {message}")
        self.send_system(f"(system notice) Tell the user now: {message}")

    def set_muted(self, muted: bool) -> None:
        self.muted = muted
        self._emit_state()

    def sleep(self) -> None:
        self._set_awake(False)

    def wake_up(self) -> None:
        self._set_awake(True)

    def restart(self, fresh: bool = False) -> None:
        """Reconnect with current settings (voice, devices). fresh=True forgets the conversation."""
        self._restart_fresh = fresh
        loaded = self._wake_name if self.wake else ""
        self._reload_wake = (self.s.wake_word or "") != loaded  # only when the wake word changed
        if self.loop and self._session_task:
            self.loop.call_soon_threadsafe(self._session_task.cancel)

    def stop(self) -> None:
        self._stopping = True
        if self.loop and self._session_task:
            self.loop.call_soon_threadsafe(self._session_task.cancel)

    # -- helpers ----------------------------------------------------------------
    def _load_wake_word(self) -> None:
        self._reload_wake = False
        self.wake = None
        if not self.s.wake_word:
            self._set_awake(True)
            return
        self.events.emit("log", text=f"Loading wake word '{self.s.wake_word}'...")
        try:
            from jarvis.wake import WakeWord

            self.wake = WakeWord(self.s.wake_word)
            self._wake_name = self.s.wake_word
            self._set_awake(False)
            self.events.emit("log", text=f"Say \"{self.s.wake_word.replace('_', ' ')}\" to wake me (or press Wake).")
        except Exception as exc:
            self.wake = None
            self.events.emit("log", text=f"[wake word off] {exc}")

    async def _send_text(self, text: str) -> None:
        if self.session:
            await self.session.send_realtime_input(text=text)

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
        self._emit_state()

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
            self.state = state
            self.events.emit("state", state=state)

    def _config(self) -> types.LiveConnectConfig:
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=self.app.prompt(),
            tools=[types.Tool(function_declarations=self.app.registry.declarations())],
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
                if not self.awake:
                    if self.wake and self.wake.heard(chunk):
                        self._set_awake(True)
                        self.events.emit("log", text="[wake word heard]")
                    continue
                if self.wake and time.monotonic() - self.last_activity > self.s.sleep_after_seconds:
                    self._set_awake(False)
                    continue
                silent = self.muted or (not self.s.allow_interrupt and (
                    speaker.speaking or time.monotonic() - self.last_spoke < ECHO_GUARD_SECONDS))
                if not silent:
                    self.events.emit("level", source="mic", value=level(chunk))
                await session.send_realtime_input(
                    audio=types.Blob(data=SILENCE if silent else chunk, mime_type=f"audio/pcm;rate={MIC_RATE}")
                )

    async def _handle_tools(self, session, call: types.LiveServerToolCall) -> None:
        responses = []
        for fc in call.function_calls or []:
            args = dict(fc.args or {})
            self.events.emit("state", state="thinking")
            result = await asyncio.to_thread(self.app.registry.call, fc.name, args, self.app.ctx)
            self.events.emit("tool", name=fc.name, args=args, result=result)
            responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=result))
        self.state = ""
        await session.send_tool_response(function_responses=responses)

    async def _receive(self, session) -> None:
        heard, said = [], []
        while True:
            async for msg in session.receive():
                if msg.session_resumption_update and msg.session_resumption_update.new_handle:
                    self.resume_handle = msg.session_resumption_update.new_handle
                if msg.go_away:
                    return  # server will close soon; reconnect with the handle
                if msg.tool_call:
                    self.last_activity = time.monotonic()
                    asyncio.create_task(self._handle_tools(session, msg.tool_call))
                sc = msg.server_content
                if not sc:
                    continue
                if sc.interrupted:
                    self.speaker.flush()
                if sc.input_transcription and sc.input_transcription.text:
                    heard.append(sc.input_transcription.text)
                    self.last_activity = time.monotonic()
                if sc.output_transcription and sc.output_transcription.text:
                    said.append(sc.output_transcription.text)
                if sc.model_turn:
                    if heard:  # show what the user said as soon as the reply starts
                        self.events.emit("user", text="".join(heard).strip())
                        heard = []
                    for part in sc.model_turn.parts or []:
                        if part.inline_data and part.inline_data.data:
                            self.speaker.play(part.inline_data.data)
                            self.last_activity = time.monotonic()
                if sc.turn_complete:
                    if heard:
                        self.events.emit("user", text="".join(heard).strip())
                    if said:
                        self.events.emit("assistant", text="".join(said).strip())
                    heard, said = [], []

    async def _session_once(self) -> None:
        if self.speaker is None:
            self.speaker = Speaker(find_device(self.s.output_device, "output"),
                                   on_level=lambda v: self.events.emit("level", source="speaker", value=v))
            self.speaker.start()
        async with self.client.aio.live.connect(model=self.s.live_model, config=self._config()) as session:
            self.session = session
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
                    self.events.emit("log", text=f"[connection] {type(exc).__name__}: {str(exc)[:150]} - retrying in {backoff}s")
                    if "handle" in str(exc).lower():
                        self.resume_handle = None  # a stale handle must not block reconnecting
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
            await asyncio.to_thread(self.background.save_summary)

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
