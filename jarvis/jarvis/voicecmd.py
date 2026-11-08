"""Understands the two commands Jarvis handles itself, without the model:
"<name>" wakes it up, and "so jao / go to sleep" puts it (never the PC) to sleep."""

from __future__ import annotations

import unicodedata

# How speech-to-text often spells "Jarvis" in English, Hindi and Urdu.
NAME_ALIASES = {
    "jarvis": ["jarvis", "jarvish", "jarwis", "jaarvis", "javis", "jervis", "jarvees",
               "जार्विस", "जारविस", "जर्विस", "जार्वीस", "جارویس", "جاروس", "جارویز"],
}

SLEEP_PHRASES = [
    "go to sleep", "sleep now", "sleep mode", "go sleep", "you can sleep", "stop listening",
    "so jao", "so ja", "sojao", "soja", "so jaao", "chup ho jao", "chup ho ja", "chup raho",
    "सो जाओ", "सो जा", "सोजाओ", "चुप हो जाओ", "चुप हो जा", "चुप रहो",
    "سو جاؤ", "سو جاو", "چپ ہو جاؤ",
]

# If any of these is in the sentence the user means the computer, so the model decides.
COMPUTER_WORDS = ["pc", "computer", "laptop", "system", "windows", "screen", "monitor",
                  "कंप्यूटर", "कम्प्यूटर", "लैपटॉप", "सिस्टम", "کمپیوٹر", "لیپ ٹاپ"]

MAX_COMMAND_WORDS = 7


def normalize(text: str) -> str:
    """Lower case, punctuation removed. Hindi/Urdu vowel signs are kept (they are marks, not punctuation)."""
    text = unicodedata.normalize("NFC", text or "").lower()
    text = "".join(" " if unicodedata.category(c)[0] in "PSZC" else c for c in text)
    return " ".join(text.split())


def _has_phrase(text: str, phrase: str) -> bool:
    """Whole-word match: the phrase's words appear one after another in the text."""
    words, want = text.split(), normalize(phrase).split()
    n = len(want)
    return n > 0 and any(words[i:i + n] == want for i in range(len(words) - n + 1))


def name_aliases(name: str) -> list[str]:
    key = normalize(name)
    return sorted({key, *NAME_ALIASES.get(key, [])} - {""}, key=len, reverse=True)


def mentions_name(text: str, name: str) -> bool:
    """True when the assistant's name was said in this bit of speech."""
    t = normalize(text)
    return any(_has_phrase(t, alias) for alias in name_aliases(name))


def is_sleep_command(text: str) -> bool:
    """A short "go to sleep" said to Jarvis. Anything about the computer is left to the model."""
    t = normalize(text)
    if not t or len(t.split()) > MAX_COMMAND_WORDS:
        return False
    if any(_has_phrase(t, w) for w in COMPUTER_WORDS):
        return False
    return any(_has_phrase(t, p) for p in SLEEP_PHRASES)
