"""Switch the language Jarvis speaks, or the language of reminders, by voice ("mujhse Hindi me baat karo")."""

from __future__ import annotations

from jarvis.lang import LANGUAGES, find, label
from jarvis.tools import tool


@tool(
    "set_language",
    "Change the language you reply in and/or the language of reminders, when the user asks to always use one "
    "('mujhse Hindi me baat karo', 'reminders English me dena', 'talk to me in Hinglish', 'auto'). "
    "for: 'jarvis' (default), 'reminders' or 'both'.",
    {"language": {"type": "string", "description": "English, Hinglish, Hindi, Urdu, Marathi, Bengali, Gujarati, "
                                                   "Punjabi, Tamil, Telugu, Spanish, French, German, Arabic, auto"},
     "for": {"type": "string", "enum": ["jarvis", "reminders", "both"]}},
    required=["language"],
)
def set_language(ctx, language: str, **kw):
    scope = (kw.get("for") or "jarvis").lower()
    if scope not in ("jarvis", "reminders", "both"):
        scope = "jarvis"
    code = find(language)
    if code is None:
        names = ", ".join(v[0].split(" · ")[0] for v in LANGUAGES.values())
        return {"error": f"I can use: {names}, or auto."}
    s = ctx.settings
    changed = []
    if scope in ("jarvis", "both"):
        if code == "same":
            return {"error": "'Same' only works for reminders."}
        s.reply_language = code
        changed.append(f"Jarvis: {label(code)}")
    if scope in ("reminders", "both"):
        s.reminder_language = "same" if scope == "both" else code
        changed.append(f"reminders: {label(s.reminder_language)}")
    from jarvis.config import save_settings

    try:
        save_settings(s)
    except OSError as exc:
        return {"error": f"Couldn't save the setting: {exc}"}
    if ctx.events is not None:
        ctx.events.emit("language", reply=s.reply_language, reminders=s.reminder_language)
    if scope == "reminders":
        how = f"Reminders will now be in {label(code)}" if code not in ("auto", "same") else \
            "Reminders will now follow your language"
        return {"result": how + ". Confirm briefly.", "saved": changed}
    how = "the user's own language each time" if code == "auto" else LANGUAGES[code][1]
    return {"result": f"Saved. From now on reply in {how}. Confirm it briefly in that language.", "saved": changed}
