# Jarvis

**A real-time voice AI assistant for your computer — by Musaddiq Khan.**

Talk to your PC in English, Hindi, Urdu or Hinglish. Jarvis listens, answers out loud and gets things done: opens and closes apps and browser tabs, controls volume, brightness and media, reads your screen and documents, sends WhatsApp messages, searches the web, checks the weather, sets reminders, writes code, and remembers what you tell it about yourself.

Powered by the Gemini Live API for low-latency, natural voice conversation, with a futuristic desktop window.

---

## Features

### The window
| | |
|---|---|
| 🔵 **Living orb with a face** | Anime-style eyes that blink and look around, smile and blush while talking, close with floating "Z z z" while asleep, look up while working. Spectrum ring, ripples when you speak, radar sweep while connecting, a flash on wake-up and a pulse + floating label for every action. |
| 🎙️ **Live captions** | What you say and what Jarvis says appear under the orb as it happens. |
| 🔊 **Sound effects** | Startup, wake, sleep, action, success, error, reminder and confirm sounds - synthesised, no files. Volume in Settings. |
| 💬 **Conversation panel** | Chat bubbles with timestamps that fade in, a chip for every action (✓ / ✗), image thumbnails (click to open) and a "working on it" indicator. |
| 📸 **Screenshots** | "Screenshot lo aur kholo" saves it to *Pictures/Jarvis Screenshots*, opens it and shows it in the chat. |
| 📂 **Drag & drop** | Drop a file or image on the window and Jarvis reads/looks at it. |
| ⌨️ **Input history** | ↑ / ↓ in the message box brings back what you typed. |
| ⚡ **Quick actions** | One-click chips: screen, weather, news, system, reminders, joke. |
| 📊 **HUD meters** | Live CPU, RAM and battery bars. |
| 🗔 **Tray + shortcuts** | Tray icon (show / wake / mute / quit). `Ctrl+M` mute · `Ctrl+Space` sleep/wake · `Ctrl+K` type · `Ctrl+L` clear · `Ctrl+,` settings. |
| ⌨️ **Type or talk** | Type in the box at any time — same session, same tools. |
| 🎙️ **Mute / 💤 Sleep** | One click to mute the mic or put Jarvis to sleep. |
| ⚠️ **Real confirmation** | Shutdown, restart, WiFi off, delete and sending messages show a banner — **you** press Confirm. |
| ⚙️ **Settings** | API key, names, voice, city, accent colour, microphone/speakers, wake word, features, auto-start. |
| 🧠 **Memory panel** | See every fact Jarvis stored about you and forget any of them. |
| 🧩 **Tools panel** | Every skill Jarvis has, including your plugins. |

### What Jarvis can do
| Area | Examples |
|---|---|
| Apps & windows | "Open Spotify" · "Close YouTube" (just the tab) · "Close Chrome" · "Switch to VS Code" · "Minimize everything" |
| Keyboard | "Press ctrl+shift+t" · "Type hello how are you and press enter" |
| System | Volume · brightness · dark mode · WiFi · lock / sleep / shutdown / restart · CPU, RAM, battery |
| Media | "Play Arijit Singh on YouTube" (plays the top result) · play/pause · next / previous |
| Vision | "What's on my screen?" · "Read this error" · "Screenshot lo aur kholo" · "Open the last screenshot" · "Look at the camera" (needs `opencv-python`) |
| Files & documents | List, read, write, move, open, create folders, delete to Recycle Bin · read PDF, Word, Excel, PowerPoint |
| Clipboard | "Translate what I copied" · "Fix the grammar of my clipboard" |
| Web | Web and news search · weather · Google Flights · Google Maps |
| Messages | WhatsApp to saved contacts (confirmed on screen) · email drafts |
| Code | "Write a Python snake game in a folder called snake" — creates the files and opens VS Code · explain/review a code file |
| Games | Open Steam / Epic downloads, library or store |
| Memory | Remembers your name, city, people, preferences, contacts · "What do you know about me?" |
| Reminders | "Remind me in 20 minutes to call mom" |
| Undo | "Undo" reverses file changes, brightness, dark mode, clipboard and saved memories |

### On its own
| | |
|---|---|
| 🌅 **Morning briefing** | First start of the day: greeting, time, weather for your city, top news, and what you talked about last time. |
| 💬 **Check-ins** | After a long quiet spell (daytime only) Jarvis makes one short, useful check-in. |
| 📊 **Hardware alerts** | Warns about CPU above 90% for 3 minutes, almost-full RAM, or battery below 20%. |
| 📰 **Watched topics** | "Watch news about Bitcoin" — Jarvis checks a few times a day and tells you about new headlines. |
| 🗓️ **Session memory** | Each conversation is summarised when you close Jarvis and mentioned in the next briefing. |
| 🗣️ **Say my name** | Asleep? Just say "Jarvis" (or whatever you named it, in any language) - "Jarvis, time kya hai?" wakes it and gets answered. |
| 💤 **"So jao"** | "So jao", "go to sleep", "chup ho jao" put **Jarvis** to sleep - never the PC. "PC ko so jao" is left to the power tool, which asks you first. Auto-sleep after a quiet spell. |
| 🔒 **Offline wake word** | Optional "Hey Jarvis" (or Alexa / Hey Mycroft) model: while asleep no audio leaves your computer. When set, it is used instead of "say my name". |
| 🚀 **Auto-start** | Optionally start with Windows. |
| 🔁 **Auto-reconnect** | Dropped connections reconnect and resume the same conversation; an expired session starts fresh instead of retrying forever. |
| 🪟 **One window** | Starting Jarvis again just brings the open window to the front. Size and position are remembered. |
| 📝 **Logs** | Everything important goes to `~/.jarvis/logs/jarvis.log` (rotated). Tray → *Open logs folder*. Crashes in any thread are logged and shown in the chat. |

---

## Quick start

Requirements: Windows 10/11 (macOS/Linux work with fewer system controls), Python 3.10+, a microphone and speakers, and a free [Gemini API key](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/musaddiq-vd/java.git
cd java/jarvis
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m jarvis                 # opens the window; paste your API key on first start
```

Double-click `jarvis_launcher.pyw` to start without a console window.

Other modes:

```bash
python -m jarvis --cli           # voice in the terminal, no window
python -m jarvis --text          # typed chat in the terminal
python -m jarvis --setup         # terminal setup (API key, names, voice)
python -m jarvis --tools         # list all tools
python -m jarvis --devices       # list microphones / speakers
```

> **Tip:** Without headphones Jarvis mutes your mic while it speaks, so it doesn't hear itself. With headphones, turn on *Let me talk over Jarvis* in Settings → Audio.

---

## Configuration

Settings live in `~/.jarvis/config.json` (or the `JARVIS_HOME` folder), **never inside the repo**, so your API key and memories can't be pushed to GitHub by mistake. Everything is editable from the Settings window.

| Key | Meaning |
|---|---|
| `api_key` | Gemini API key (the `GEMINI_API_KEY` env var overrides it) |
| `assistant_name`, `user_name`, `city` | Names and city used in conversation and the briefing |
| `voice` | `Charon`, `Puck`, `Kore`, `Fenrir` or `Aoede` |
| `live_model`, `text_model` | Gemini models for voice and for text work |
| `theme_color` | Accent colour of the window |
| `input_device`, `output_device` | Microphone and speakers by name; empty = default |
| `allow_interrupt` | Talk over Jarvis (headphones) |
| `wake_word`, `sleep_after_seconds` | Offline wake word model and auto-sleep time |
| `name_wake` | Wake when you say the assistant's name (used when no offline wake word is set) |
| `sound_effects`, `sound_volume` | Sound effects on/off and their volume (0-100) |
| `orb_face` | The orb's animated face |
| `morning_briefing`, `proactive`, `proactive_minutes`, `hardware_alerts` | Background features |
| `allowed_roots` | Folders the file tools may use; empty = your home folder |

Memories: `~/.jarvis/memory.json` · app state (watched topics, last summary): `~/.jarvis/state.json` · logs: `~/.jarvis/logs/`.

---

## Writing a plugin

Create `plugins/hello.py`:

```python
from jarvis.tools import tool

@tool("say_hello", "Greet someone by name.",
      {"name": {"type": "string"}}, required=["name"])
def say_hello(ctx, name):
    return f"Hello, {name}!"
```

Restart Jarvis and say *"say hello to Ali"*. `ctx` gives you `settings`, `memory`, `undo`, `confirm`, `notify`, `state`, `llm` and `attach_image`. See [`plugins/_example.py`](plugins/_example.py). Plugins can also go in `~/.jarvis/plugins/`.

---

## Safety by design

- No `exec()` of model output and no `shell=True` anywhere.
- File tools only work inside allowed folders; deletes go to the Recycle Bin.
- Irreversible actions need a click from you — the model can't confirm them itself.
- Keyboard shortcuts are limited to a known key list; URLs must be http(s).
- Secrets and personal data stay in `~/.jarvis`, outside the repository.

---

## Project structure

```
jarvis/
├── jarvis/
│   ├── cli.py            # entry point (window by default)
│   ├── app.py            # wires settings, memory, tools, events and the system prompt
│   ├── live.py           # Gemini Live voice session (mic, speaker, tools, vision, reconnect)
│   ├── background.py     # briefing, check-ins, hardware alerts, watched topics, summaries
│   ├── text_chat.py      # typed chat with the same tools
│   ├── llm.py            # text model with automatic fallback
│   ├── events.py         # event bus between core and window
│   ├── voicecmd.py       # "Jarvis" wakes it, "so jao" puts it to sleep
│   ├── sfx.py            # synthesised sound effects
│   ├── logs.py           # log file + crash capture
│   ├── audio.py · wake.py · net.py
│   ├── config.py · memory.py · state.py · undo.py · confirm.py
│   ├── tools/            # built-in tools, one file per area
│   └── ui/               # PyQt6 window, orb, settings
├── plugins/              # your own tools
├── tests/                # pytest suite
└── jarvis_launcher.pyw    # start without a console
```

## Development

```bash
pip install pytest
python -m pytest
```

---

## Copyright

© 2026 Musaddiq Khan. All rights reserved.
