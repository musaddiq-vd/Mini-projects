from types import SimpleNamespace

from google.genai import errors, types

from jarvis.config import Settings
from jarvis.llm import TextModel
from jarvis.text_chat import TextChat


class FakeModels:
    def __init__(self, behaviour):
        self.behaviour, self.calls = behaviour, []

    def generate_content(self, model, contents, config=None):
        self.calls.append(model)
        result = self.behaviour.get(model)
        if callable(result):
            result = result(contents)
        if isinstance(result, Exception):
            raise result
        return result


def _reply(text=None, call=None):
    part = types.Part(function_call=types.FunctionCall(name=call[0], args=call[1])) if call else types.Part(text=text)
    content = types.Content(role="model", parts=[part])
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


def _model(behaviour):
    m = TextModel(Settings(api_key="x"), log=lambda *_: None)
    m.models = ["broken-model", "good-model"]
    m.client = SimpleNamespace(models=FakeModels(behaviour))
    return m


def _chat(ctx, registry, behaviour):
    app = SimpleNamespace(settings=ctx.settings, registry=registry, ctx=ctx, prompt=lambda: "test")
    return TextChat(app, model=_model(behaviour))


def test_falls_back_to_next_model(ctx, registry):
    retired = errors.ClientError(404, {"error": {"code": 404, "message": "gone", "status": "NOT_FOUND"}})
    chat = _chat(ctx, registry, {"broken-model": retired, "good-model": _reply("hi")})
    assert chat.send("hello") == "hi"
    assert chat.send("again") == "hi"
    assert chat.model.client.models.calls == ["broken-model", "good-model", "good-model"]


def test_bad_request_is_not_retried_and_history_is_cleaned(ctx, registry):
    bad = errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})
    chat = _chat(ctx, registry, {"broken-model": bad})
    try:
        chat.send("hello")
    except errors.ClientError:
        pass
    assert chat.model.client.models.calls == ["broken-model"]
    assert chat.history == []


def test_tool_round_and_image_attachment(ctx, registry, monkeypatch):
    import jarvis.tools.vision as vision

    monkeypatch.setattr(vision, "screenshot_jpeg", lambda monitor=1: b"JPEG")

    def good(contents):
        if len(contents) == 1:
            return _reply(call=("look_at_screen", {}))
        last = contents[-1]
        assert any(p.inline_data and p.inline_data.data == b"JPEG" for p in last.parts)
        return _reply("I see your screen")

    chat = _chat(ctx, registry, {"broken-model": good})
    tools_seen = []
    assert chat.send("what's on my screen", on_tool=lambda n, a, r: tools_seen.append(n)) == "I see your screen"
    assert tools_seen == ["look_at_screen"]
