"""Voice profiles: learn a person's voice, tell who is speaking, and use their own settings.

Only a small set of numbers per person is kept (~/.jarvis/voices/), never the recordings.
See jarvis/voices.py for how it works and how accurate it is.
"""

from __future__ import annotations

import time

from jarvis import voices
from jarvis.tools import tool

PAUSE = 0.6  # between phrases
WAIT_FOR_SILENCE = 12.0  # wait at most this long for Jarvis to finish talking before recording


def record_phrase(ctx, seconds: float):
    """Record one phrase from the microphone (16 kHz mono int16). Replaced in tests."""
    import sounddevice as sd

    from jarvis.audio import find_device

    data = sd.rec(int(seconds * voices.RATE), samplerate=voices.RATE, channels=1, dtype="int16",
                  device=find_device(ctx.settings.input_device, "input"))
    sd.wait()
    return data[:, 0].copy()


def _quiet_assistant(ctx):
    """Wait until Jarvis has stopped talking (so the recording isn't its own voice) and mute the live mic."""
    assistant = getattr(ctx, "assistant", None)
    if assistant is None:
        return None
    end = time.monotonic() + WAIT_FOR_SILENCE
    speaker = getattr(assistant, "speaker", None)
    while speaker is not None and speaker.speaking and time.monotonic() < end:
        time.sleep(0.1)
    was = assistant.muted
    assistant.set_muted(True)
    return was


@tool(
    "enroll_voice",
    "Learn a person's voice so Jarvis knows who is speaking ('learn my voice', 'meri awaaz pehchano', 'add my "
    "wife's voice as Sara'). Before calling, say: 'Read the three short lines on the screen when they appear.' "
    "The person reads 3 short phrases (about 15 seconds). Optional per-person settings: language (code like "
    "'hi', 'en' or 'auto'), speaking_style (friendly, calm, energetic, professional, funny), memory ('all' "
    "facts, 'own' = only facts about them, 'none'), owner=true for the main user.",
    {"name": {"type": "string"}, "language": {"type": "string"}, "speaking_style": {"type": "string"},
     "memory": {"type": "string", "enum": list(voices.MEMORY_CHOICES)}, "owner": {"type": "boolean"}},
    required=["name"], slow=True,
)
def enroll_voice(ctx, name: str, language: str = "auto", speaking_style: str = "", memory: str = "",
                 owner: bool | None = None):
    name = (name or "").strip()
    if not name:
        return {"error": "Whose voice is it? I need a name."}
    if voices.active().get("guest") and voices.load_profiles():
        if not ctx.confirm.confirm(f"An unknown voice wants to save a voice profile for '{name}'. Allow it?"):
            return {"result": "Cancelled - the owner didn't allow a new voice profile."}
    was_muted = _quiet_assistant(ctx)
    samples = []
    try:
        for i, phrase in enumerate(voices.PHRASES, 1):
            if ctx.events is not None:
                ctx.events.emit("voice_enroll", step=i, total=len(voices.PHRASES), phrase=phrase, name=name)
                ctx.events.emit("log", text=f"🎙 {name}, please read ({i}/{len(voices.PHRASES)}): “{phrase}”")
            time.sleep(PAUSE)
            samples.append(record_phrase(ctx, voices.PHRASE_SECONDS))
    except Exception as exc:
        return {"error": f"Couldn't record from the microphone: {type(exc).__name__}: {exc}"}
    finally:
        if was_muted is not None:
            ctx.assistant.set_muted(was_muted)
        if ctx.events is not None:
            ctx.events.emit("voice_enroll", step=0, total=len(voices.PHRASES), phrase="", name=name)
    heard = sum(1 for s in samples if voices.voiced_seconds(s) >= voices.MIN_VOICED_SECONDS)
    if heard < 2:
        return {"error": "I couldn't hear enough speech. Ask them to read the lines a bit louder, closer to the "
                         "microphone, then try again."}
    try:
        profile = voices.enroll(name, samples, language=language, style=speaking_style, memory=memory, owner=owner)
    except ValueError:
        return {"error": "I couldn't hear enough speech - please try again in a quieter spot."}
    finally:
        samples.clear()  # the audio is never kept
    watcher = getattr(getattr(ctx, "assistant", None), "voice_watch", None)
    if watcher is not None:
        watcher.reload()
    if ctx.events is not None:
        ctx.events.emit("voice_profiles")
    role = "the owner" if profile["owner"] else "a family member / friend"
    return {"result": f"Saved {name}'s voice as {role} ({heard} of {len(voices.PHRASES)} phrases heard). "
                      "Only voice numbers are kept, not the recording. Recognition is a convenience, not a "
                      "security check; it works best in a quiet room.",
            "name": name, "owner": profile["owner"]}


@tool("who_is_speaking", "Whose voice Jarvis heard last ('who am I?', 'main kaun hoon?'), from the voice profiles.")
def who_is_speaking(ctx):
    profiles = voices.load_profiles()
    if not profiles:
        return {"result": "No voice profiles yet - say 'learn my voice' to add one."}
    if not getattr(ctx.settings, "voice_id", True):
        return {"result": "Voice recognition is off in Settings > Privacy."}
    a = voices.active()
    if not a:
        return {"result": "I haven't recognised anyone yet - say a sentence or two first."}
    seconds = int(time.time() - a.get("at", time.time()))
    if a.get("guest"):
        return {"speaker": "guest", "result": "The last voice I heard doesn't match a saved profile (a guest).",
                "seconds_ago": seconds}
    return {"speaker": a["name"], "seconds_ago": seconds,
            "result": f"It sounds like {a['name']} (by voice - not a security check)."}


@tool("list_voice_profiles", "The saved voice profiles with each person's settings (owner, language, style, "
                             "which memory facts).")
def list_voice_profiles(ctx):
    profiles = voices.load_profiles()
    if not profiles:
        return {"result": "No voice profiles yet."}
    return {"profiles": [{"name": p["name"], "owner": bool(p.get("owner")), "language": p.get("language", "auto"),
                          "style": p.get("style") or "default", "memory": p.get("memory", "all")}
                         for p in profiles],
            "recognition_on": bool(getattr(ctx.settings, "voice_id", True)),
            "guests_may_use_private_tools": bool(getattr(ctx.settings, "guest_private_tools", False))}


@tool("delete_voice_profile", "Delete a person's voice profile (the user confirms on screen).",
      {"name": {"type": "string"}}, required=["name"])
def delete_voice_profile(ctx, name: str):
    p = voices.get_profile(name)
    if p is None:
        return {"error": f"There's no voice profile called '{name}'."}
    if voices.is_guest():
        return {"error": "Only a recognised voice (or Settings > Privacy) can delete voice profiles."}
    if not ctx.confirm.confirm(f"Delete {p['name']}'s voice profile?"):
        return {"result": "Cancelled - nothing was deleted."}
    voices.delete_profile(p["name"])
    watcher = getattr(getattr(ctx, "assistant", None), "voice_watch", None)
    if watcher is not None:
        watcher.reload()
    if ctx.events is not None:
        ctx.events.emit("voice_profiles")
    return {"result": f"Deleted {p['name']}'s voice profile."}
