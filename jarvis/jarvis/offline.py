"""Offline mode: everyday commands still work when there's no internet.

parse("5 minute ka timer") -> Command("set_timer", {"minutes": 5}) for: timers, reminders, notes and to-dos,
"my list", opening apps, volume, the time and how much is left on a timer. English and Hinglish.

OfflineVoice speaks replies with the Windows voice (no internet needed) and, while offline, listens with the
built-in Windows speech recognizer (English; say the assistant's name first: "Jarvis, set a timer for 5 minutes").
"""

from __future__ import annotations

import logging
import platform
import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

log = logging.getLogger("jarvis.offline")

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty five": 45,
                "sixty": 60, "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "das": 10,
                "pandrah": 15, "bees": 20, "tees": 30, "half an": 0.5, "a": 1, "an": 1}
MIN = r"(?:minutes?|mins?|minute|minat|mint)"
SEC = r"(?:seconds?|secs?|second)"
HOUR = r"(?:hours?|hrs?|ghante|ghanta)"


@dataclass
class Command:
    tool: str
    args: dict = field(default_factory=dict)


def _num(text: str) -> float | None:
    t = text.strip().lower()
    try:
        return float(t)
    except ValueError:
        return NUMBER_WORDS.get(t)


def _clean(text: str, name: str = "jarvis") -> str:
    t = " ".join(text.strip().split())
    t = re.sub(r"^(?:hey |ok |okay )?(?:jarvis|" + re.escape(name) + r")[,!. ]*", "", t, flags=re.I)  # spoken name
    return t.strip(" .!?")


def _duration(t: str) -> tuple[float, float] | None:
    """'5 minute', '1 hour 10 min', '30 seconds', 'five minutes' -> (minutes, seconds)."""
    num = r"(\d+(?:\.\d+)?|[a-z]+(?: [a-z]+)?)"
    h = re.search(num + r"\s*" + HOUR, t)
    m = re.search(num + r"\s*" + MIN, t)
    s = re.search(num + r"\s*" + SEC, t)
    minutes = seconds = 0.0
    found = False
    for match, mult, kind in ((h, 60, "m"), (m, 1, "m"), (s, 1, "s")):
        if match:
            words = match.group(1).split()
            value = _num(match.group(1)) or (_num(words[-1]) if words else None)
            if value is None:
                continue
            found = True
            if kind == "m":
                minutes += value * mult
            else:
                seconds += value
    return (minutes, seconds) if found else None


def parse(text: str, name: str = "jarvis") -> Command | None:
    """A command we can run without the internet, or None."""
    raw = _clean(text, name)
    t = raw.lower()
    if not t:
        return None
    # -- timers
    if "timer" in t and not re.search(r"\b(cancel|band|stop|hatao)\b", t):
        if re.search(r"(kitna|how much|left|bacha|baaki|remaining)", t):
            return Command("list_timers")
        d = _duration(t)
        if d:
            label = re.search(r"\bfor (?:the |my )?([a-z ]+?)(?: for| timer|$)", t)
            lab = label.group(1).strip() if label and not re.search(MIN + "|" + SEC, label.group(1)) else ""
            return Command("set_timer", {"minutes": d[0], "seconds": d[1], "label": lab})
    # -- reminders
    m = re.match(r"remind me (?:in|after) (.+?) (?:to|that|about) (.+)$", t)
    if m and _duration(m.group(1)):
        mins, secs = _duration(m.group(1))
        return Command("set_reminder", {"message": raw[m.start(2):], "minutes": max(1 / 60, mins + secs / 60)})
    m = re.match(r"remind me (?:at|by) ([\d:. ]+(?:am|pm)?) (?:to|that|about) (.+)$", t)
    if m:
        return Command("set_reminder", {"message": raw[m.start(2):], "at": m.group(1).strip()})
    m = re.match(r"remind me to (.+?) (?:in|after) (.+)$", t)
    if m and _duration(m.group(2)):
        mins, secs = _duration(m.group(2))
        return Command("set_reminder", {"message": raw[m.start(1):m.end(1)], "minutes": max(1 / 60, mins + secs / 60)})
    if re.search(r"yaad dila", t):
        body = re.sub(r"\b(?:mujhe|please|plz)\b", "", t)
        d_match = re.search(r"(\d+|[a-z]+)\s*(?:" + MIN + "|" + HOUR + r")\s*(?:baad|me|mein)", body)
        clock = re.search(r"(\d{1,2}(?::\d{2})?)\s*baje", body)
        what = re.sub(r"(\d+|[a-z]+)\s*(?:" + MIN + "|" + HOUR + r")\s*(?:baad|me|mein)|\d{1,2}(?::\d{2})?\s*baje"
                      r"|(?:shaam|subah|raat|dopahar)\s*|yaad dila(?:na|do|dena|iye|o)?|,", " ", body)
        what = " ".join(what.split())
        what = re.sub(r"^(?:ki|ke|that|to)\s+|\s+(?:ki|ke)$", "", what).strip()  # only the joining words
        if what and d_match:
            d = _duration(d_match.group(0).replace("baad", "").replace("mein", "").replace(" me", ""))
            if d:
                return Command("set_reminder", {"message": what, "minutes": max(1 / 60, d[0] + d[1] / 60)})
        if what and clock:
            part = re.search(r"(shaam|subah|raat|dopahar)", body)
            return Command("set_reminder", {"message": what, "at": f"{part.group(1) + ' ' if part else ''}{clock.group(1)}"})
    # -- lists (adding before showing: "add eggs to my list" mentions "my list" too)
    m = re.match(r"(?:add|put) (.+?) (?:to|on|in) (?:my |the )?(?:to-?do )?list$", t)
    if m:
        return Command("add_note", {"text": raw[m.start(1):m.end(1)], "todo": True})
    m = re.match(r"(.+?),? (?:ko )?(?:list|to-?do list) (?:me|mein|pe|par) (?:daal|dal|add kar|likh)(?: do| dena| de|o)?$", t)
    if m:
        return Command("add_note", {"text": raw[m.start(1):m.end(1)], "todo": True})
    m = re.match(r"(?:note karo|note kar lo|note down|take a note|note)[:,]? (.+)$", t)
    if m:
        return Command("add_note", {"text": raw[m.start(1):], "todo": False})
    if re.search(r"\b(meri list|my list|to-?do list|todo list|list batao|show (?:my )?list)\b", t):
        return Command("list_notes")
    if re.search(r"\b(my reminders|mere reminders|reminders? (?:batao|dikhao)|what reminders)\b", t):
        return Command("list_reminders")
    # -- smart home (Tuya devices work on the home WiFi, no internet needed)
    m = re.match(r"(?:turn|switch) (on|off) (?:the )?(.+)$", t) or re.match(r"(?:turn|switch) (?:the )?(.+?) (on|off)$", t)
    if m:
        on_off, device = (m.group(1), m.group(2)) if m.group(1) in ("on", "off") else (m.group(2), m.group(1))
        if not re.search(r"\b(wifi|bluetooth|computer|pc|laptop|volume|dark mode)\b", device):
            return Command("smart_home", {"device": device, "action": on_off})
    m = re.match(r"(.+?) (on|off|chalu|band) (?:karo|kar do|kardo|karna)$", t)
    if m and not re.search(r"\b(wifi|bluetooth|computer|pc|laptop|volume|timer|music|gaana)\b", m.group(1)):
        action = "on" if m.group(2) in ("on", "chalu") else "off"
        return Command("smart_home", {"device": m.group(1), "action": action})
    # -- volume
    if "volume" in t or "awaaz" in t or "awaz" in t:
        if re.search(r"\b(mute|band)\b", t):
            return Command("volume", {"action": "mute"})
        if re.search(r"\b(up|badhao|badha do|zyada|increase|louder|tez)\b", t):
            return Command("volume", {"action": "up"})
        if re.search(r"\b(down|kam|ghatao|decrease|lower|dheere)\b", t):
            return Command("volume", {"action": "down"})
    # -- time
    if re.search(r"\b(what time|time kya|kitne baje|time batao|what's the time|current time|samay)\b", t):
        return Command("get_time")
    # -- open apps
    m = re.match(r"(?:open|launch|start) (.+)$", t) or re.match(r"(.+?) (?:kholo|khol do|open karo|open kar do|chalu karo)$", t)
    if m:
        name = m.group(1).strip()
        if name and len(name) <= 40 and not re.search(r"\b(website|\.com|http)\b", name):
            return Command("open_app", {"name": name})
    return None


def reply(cmd: Command, result: dict) -> str:
    """A short spoken / shown answer for a tool result."""
    if "error" in result:
        return f"Sorry - {result['error']}"
    if cmd.tool == "list_notes" and isinstance(result.get("items"), list):
        items = [i["text"] for i in result["items"] if i.get("type") == "todo"][:6]
        return ("Your list: " + ", ".join(items) + ".") if items else "Your to-do list is empty."
    if cmd.tool == "list_reminders" and isinstance(result.get("reminders"), list):
        rs = result["reminders"][:5]
        return "Reminders: " + "; ".join(f"{r['message']} at {r['at']}" for r in rs) + "." if rs else "No reminders."
    if cmd.tool == "get_time":
        return str(result.get("result") or result.get("time") or result)
    text = result.get("result")
    if isinstance(text, str):
        return text
    return "Done."


# -- the Windows voice, without internet ----------------------------------------------------------
_SPEAK_PS = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = 1
while ($true) {
  $line = [Console]::In.ReadLine()
  if ($line -eq $null) { break }
  if ($line.Length -gt 0) { try { $s.Speak($line) } catch {} }
}
"""
_LISTEN_PS = r"""
Add-Type -AssemblyName System.Speech
try {
  $r = New-Object System.Speech.Recognition.SpeechRecognitionEngine
  $r.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
  $r.SetInputToDefaultAudioDevice()
} catch { [Console]::Out.WriteLine("NOREC:" + $_.Exception.Message); [Console]::Out.Flush(); exit 1 }
[Console]::Out.WriteLine("READY"); [Console]::Out.Flush()
while ($true) {
  $res = $r.Recognize([TimeSpan]::FromSeconds(8))
  if ($res -ne $null) { [Console]::Out.WriteLine("HEARD:" + $res.Confidence + ":" + $res.Text); [Console]::Out.Flush() }
}
"""
NO_WINDOW = 0x08000000


class OfflineVoice:
    """Windows' own speech engine in two small background processes (speak / listen)."""

    def __init__(self, on_heard: Callable[[str], None] | None = None, name: str = "Jarvis"):
        self.on_heard = on_heard
        self.name = name.lower()
        self._speaker: subprocess.Popen | None = None
        self._listener: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.available = platform.system() == "Windows"

    def _ps(self, script: str) -> subprocess.Popen:
        return subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, encoding="utf-8", creationflags=NO_WINDOW)

    def say(self, text: str) -> None:
        if not self.available or not text:
            return
        line = " ".join(re.sub(r"[^\w\s,.!?'%:-]", " ", text).split())[:400]
        with self._lock:
            try:
                if self._speaker is None or self._speaker.poll() is not None:
                    self._speaker = self._ps(_SPEAK_PS)
                self._speaker.stdin.write(line + "\n")
                self._speaker.stdin.flush()
            except OSError as exc:
                log.warning("offline voice failed: %s", exc)
                self._speaker = None

    def listen(self) -> None:
        """Start listening (only commands that begin with the assistant's name are passed on)."""
        if not self.available or self.on_heard is None:
            return
        with self._lock:
            if self._listener is not None and self._listener.poll() is None:
                return
            try:
                self._listener = self._ps(_LISTEN_PS)
            except OSError as exc:
                log.warning("offline listening failed: %s", exc)
                return
            proc = self._listener
        threading.Thread(target=self._read, args=(proc,), daemon=True, name="jarvis-offline-listen").start()

    def _read(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("NOREC:"):
                log.warning("no offline speech recognizer: %s", line[6:])
                return
            if not line.startswith("HEARD:"):
                continue
            _, conf, said = line.split(":", 2)
            try:
                if float(conf) < 0.4:
                    continue
            except ValueError:
                continue
            if addressed(said, self.name) and self.on_heard:
                self.on_heard(said)

    def stop_listening(self) -> None:
        with self._lock:
            proc, self._listener = self._listener, None
        if proc is not None and proc.poll() is None:
            proc.kill()

    def close(self) -> None:
        self.stop_listening()
        with self._lock:
            proc, self._speaker = self._speaker, None
        if proc is not None and proc.poll() is None:
            proc.kill()


def addressed(said: str, name: str = "jarvis") -> bool:
    """Only 'Jarvis, ...' counts, so a TV or a conversation in the room can't give commands."""
    words = re.findall(r"[a-z]+", said.lower())
    return bool(words) and (words[0] == name.lower() or (len(words) > 1 and words[0] in ("hey", "ok", "okay")
                                                        and words[1] == name.lower()))
