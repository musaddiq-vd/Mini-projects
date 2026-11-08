"""Human-like speaking: styles in the prompt, voices, expressive-voice fallback."""

from jarvis.app import create_app, system_prompt
from jarvis.config import SPEAKING_STYLES, VOICE_INFO, VOICES, Settings


def test_every_style_reaches_the_prompt(ctx):
    for key, desc in SPEAKING_STYLES.items():
        ctx.settings.speaking_style = key
        text = system_prompt(ctx.settings, ctx.memory)
        assert desc in text and "VOICE call" in text and "as an AI" in text


def test_unknown_style_falls_back(ctx):
    ctx.settings.speaking_style = "pirate"
    assert SPEAKING_STYLES["friendly"] in system_prompt(ctx.settings, ctx.memory)


def test_voices_keep_the_original_five():
    assert VOICES[:5] == ("Charon", "Puck", "Kore", "Fenrir", "Aoede") and len(VOICE_INFO) >= 30


def test_expressive_voice_falls_back_and_is_remembered(isolated_home):
    from jarvis.live import LiveAssistant

    app = create_app(Settings(api_key="k", sound_effects=False))
    a = LiveAssistant(app)
    assert a._config().enable_affective_dialog is True
    a._on_connection_error(Exception("1007 None. Request contains an invalid argument."), 1)
    assert a._affective_ok is False and a._config().enable_affective_dialog is None
    again = LiveAssistant(create_app(Settings(api_key="k", sound_effects=False)))
    assert again._affective_ok is False  # remembered for this model


def test_expressive_voice_off_in_settings(isolated_home):
    from jarvis.live import LiveAssistant

    a = LiveAssistant(create_app(Settings(api_key="k", sound_effects=False, expressive_voice=False)))
    assert a._config().enable_affective_dialog is None
    a._on_connection_error(Exception("invalid argument"), 1)
    assert a._affective_ok  # not blamed when it wasn't even on
