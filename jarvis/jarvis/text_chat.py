"""Typed chat with the same tools - handy for testing without a microphone."""

from __future__ import annotations

from google.genai import types

from jarvis.app import Jarvis
from jarvis.llm import TextModel

MAX_TOOL_ROUNDS = 8


class TextChat:
    def __init__(self, app: Jarvis, model: TextModel | None = None):
        self.app = app
        self.model = model or TextModel(app.settings)
        self.history: list[types.Content] = []
        self._images: list[bytes] = []
        app.ctx.attach_image = self._images.append

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=self.app.prompt(),
            tools=[types.Tool(function_declarations=self.app.registry.declarations())],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def send(self, text: str, on_tool=None) -> str:
        start = len(self.history)
        self.history.append(types.Content(role="user", parts=[types.Part(text=text)]))
        try:
            return self._run_turn(on_tool)
        except Exception:
            del self.history[start:]  # drop the half-finished turn so history stays valid
            raise

    def _run_turn(self, on_tool) -> str:
        config = self._config()
        for _ in range(MAX_TOOL_ROUNDS):
            response = self.model.generate(self.history, config)
            candidate = response.candidates[0] if response.candidates else None
            if candidate is None or candidate.content is None:
                raise RuntimeError("the model returned no reply")
            self.history.append(candidate.content)
            calls = [p.function_call for p in candidate.content.parts or [] if p.function_call]
            if not calls:
                return response.text or ""
            parts = []
            for call in calls:
                args = dict(call.args or {})
                result = self.app.registry.call(call.name, args, self.app.ctx)
                if on_tool:
                    on_tool(call.name, args, result)
                parts.append(types.Part.from_function_response(name=call.name, response=result))
            parts += [types.Part.from_bytes(data=img, mime_type="image/jpeg") for img in self._images]
            self._images.clear()
            self.history.append(types.Content(role="user", parts=parts))
        return "(stopped after too many tool calls)"


def run_text_chat(app: Jarvis) -> None:
    chat = TextChat(app)
    name = app.settings.assistant_name
    print(f"{name} text mode. Type 'exit' to quit.\n")
    while True:
        try:
            text = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text.lower() in {"exit", "quit", "bye"}:
            return
        try:
            reply = chat.send(text, on_tool=lambda n, a, r: print(f"   [tool] {n} {a}"))
        except Exception as exc:
            reply = f"(error: {str(exc)[:300]})"
        print(f"{name.lower()} > {reply}\n")
