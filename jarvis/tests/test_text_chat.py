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


def test_a_failed_model_is_skipped_after_a_restart(ctx, registry, monkeypatch):
    import time

    from jarvis import llm

    quota = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    chat = _chat(ctx, registry, {"broken-model": quota, "good-model": _reply("hi")})
    assert chat.send("hello") == "hi"
    # a new start: the quota-hit model goes to the back, so nobody waits for it again
    assert llm.usable_order(["broken-model", "good-model"]) == ["good-model", "broken-model"]
    later = time.time() + 31 * 60  # the cool-down is over: back in front
    assert llm.usable_order(["broken-model", "good-model"], now=later) == ["broken-model", "good-model"]
    llm.mark_failed("gone-model", 404)
    assert llm.usable_order(["gone-model", "x"], now=time.time() + 3600) == ["x", "gone-model"]


def test_the_last_model_gets_one_more_try_when_busy(ctx, registry, monkeypatch):
    from jarvis import llm

    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    busy = errors.ServerError(504, {"error": {"code": 504, "message": "deadline", "status": "DEADLINE_EXCEEDED"}})
    answers = iter([busy, _reply("hi")])
    chat = _chat(ctx, registry, {"good-model": lambda contents: next(answers)})
    chat.model.models = ["good-model"]
    assert chat.send("hello") == "hi" and chat.model.client.models.calls == ["good-model", "good-model"]


def test_all_models_out_of_quota_gives_a_clear_message(ctx, registry):
    from jarvis.llm import QuotaReached

    quota = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    chat = _chat(ctx, registry, {"broken-model": quota, "good-model": quota})
    try:
        chat.send("hello")
        raise AssertionError("expected QuotaReached")
    except QuotaReached as exc:
        assert "usage limit" in str(exc)
    assert chat.history == []
