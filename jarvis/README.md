# Jarvis

**A real-time voice AI assistant for your computer — by Musaddiq Khan.**

Talk to your PC in English, Hindi, Urdu or Hinglish. Jarvis listens, answers out loud and gets things done: opens and closes apps and browser tabs, controls volume, brightness and media, reads your screen and documents, sends WhatsApp messages, searches the web, checks the weather, sets reminders, writes code, and remembers what you tell it about yourself.

Powered by the Gemini Live API for low-latency, natural voice conversation, with a futuristic desktop window.

---

## Features

### The window
| | |
|---|---|
| 🫧 **Cute moments** | On great moments the AI core morphs for a few seconds into a bubbly anime face with a burst of sparkles, then back: heart eyes when you say thanks, a laughing face for "wow / mast / zabardast", star eyes when Jarvis congratulates you, a wink after a joke - and now and then a tiny surprise while idle. |
| 💖 **Feelings** | LED heart eyes and floating hearts when you say thanks, a wink after a joke, tilted dim visors with a tear when something fails, surprised rings when a timer rings. |
| ⏱️ **Timer rings** | Every running timer is a countdown ring around the orb with a label; it blinks in the last seconds and an alarm plays. |
| 🧠 **AI core** | A glass sphere with a rotating hex lattice, a scan line and glowing LED visor eyes that blink and glance around. Talking turns them into happy arcs with a live voice waveform for a mouth; working shows scanner lights sweeping in the visors; asleep they dim with floating "Z z z". Around it: a neural ring of nodes that light up with sound, a spectrum ring, ripples when you speak, a radar sweep while connecting, a flash on wake-up and a pulse + floating label for every action. |
| ✨ **Motion** | Boot animation ("INITIALIZING") on start, a slowly drifting aurora + grid background, glass panels, buttons and the message box glow on hover/focus, chat rows grow and fade in, Jarvis's replies type themselves out, toast cards slide in for reminders, timers and connection changes. |
| 🎙️ **Live captions** | What you say and what Jarvis says appear under the orb as it happens. |
| 🔊 **Sound effects** | Startup, wake, sleep, action, success, error, reminder and confirm sounds - synthesised, no files. Volume in Settings. |
| 💬 **Conversation panel** | Chat bubbles with timestamps that fade in, a chip for every action (✓ / ✗), image thumbnails (click to open) and a "working on it" indicator. |
| 📸 **Screenshots** | "Screenshot lo aur kholo" saves it to *Pictures/Jarvis Screenshots*, opens it and shows it in the chat. |
| 📂 **Drag & drop** | Drop a file or image on the window and Jarvis reads/looks at it. |
| 🏆 **Achievements** | 19 badges you unlock just by using Jarvis (Early Bird, Night Owl, Timer Master, List Maker, Week Warrior, Perfect Score...). Unlocking one bursts confetti with a fanfare; the 🏆 button shows them all with progress bars. |
| 🌦 **Rich cards** | Weather shows as a card (big temperature, condition, high/low, rain, humidity, wind); web and news results as clickable link cards (only http/https links, with their site shown). |
| 🌙 **Evening recap** | In the evening (from 9 pm by default) - or any time you say "aaj ka recap do" - Jarvis sums up your day: what you got done, what's open, what's coming, badges, practice; then asks about tomorrow. |
| 💧 **Health nudges** | While you use the PC: a 20-20-20 eye break and stretch every 50 minutes and water every 90 (stepping away resets the clock; quiet at night, in focus and practice mode). Only "seconds since last key/mouse input" is read - nothing is recorded. |
| 💰 **Expenses** | "100 ki chai pi", "kal 450 Uber" -> saved by category; "is mahine kitna kharcha hua?"; monthly budgets with warnings at 80% and 100%; undo. |
| ✅ **Habits** | "Gym ko daily habit bana do", "2 glass paani piya" -> daily or weekday habits with targets, streaks (weekends don't break weekday habits) and a 7-day view. |
| 🔎 **Smart file search** | "Kal wala PDF jisme invoice tha" -> searches names *and* the text inside PDFs/Word/Excel/PowerPoint, filtered by type and date. Only in your allowed folders; secret folders are never searched. |
| 📄 **Document chat** | Drop a file on the window (or ask Jarvis to find one) and ask "total kitna hai?", "summary do" - answers come only from the document, with page numbers. Scanned and password-protected files are explained, not guessed. |
| 📅 **Calendar** | Paste your private iCal link (Google / Outlook / Apple) in Settings > Features. "Aaj meri meetings?", "kal free kab hu?", "next meeting?"; a spoken reminder 10 min before each meeting (with the join link), and the next meeting on the Today strip. Read-only, works offline from the last copy; links are encrypted and never logged. |
| 🎯 **Goal coach** | "Mujhe 2 mahine mein 5 km daudna hai" -> Jarvis plans small dated steps until the deadline, tracks progress and follows up once a day on the step that's due. |
| 📊 **Dashboard** | `Ctrl+D` / 📊 / "dashboard dikhao": your week (messages, tasks, focus minutes, spending), habits with +1 buttons, money with budget bar and charts, goals with tickable steps. |
| 📋 **Planner** | `Ctrl+P` or the 📋 button: tick and add to-dos, add or cancel reminders ("6:30 pm" / "20 min"), quick 1/5/10/25-minute timers and a focus session, run or delete routines. Guardrails and undo apply here too. |
| 📶 **Connection pill** | Under the clock: Online (with the model), Connecting, Reconnecting or Offline at a glance. |
| 📅 **Today strip** | Under the orb: next reminder, running timer, open to-dos, streak and badges - click a chip to ask about it. |
| 🗣 **Voice control of the window** | "Night mode karo", "Sunset theme lagao", "chhote ho jao", "help dikhao", "mere achievements dikhao". |
| ❓ **What can I say?** | `F1` or the **?** button opens example commands by topic - click one to try it. An empty chat shows a welcome card with ideas. |
| 🔒 **Privacy tab** | Settings › Privacy shows everything Jarvis keeps (with sizes), deletes any of it, sets history retention, asks before the camera, and flashes the window whenever the screen or camera is used. |
| ⌨️ **Input history** | ↑ / ↓ in the message box brings back what you typed. |
| 🌓 **Day / night themes** | Six themes (Jarvis Blue, Neon Pink, Emerald, Sunset, Royal Purple, Mono) in dark, light or auto mode (light by day, dark at night). The 🌙/☀ button in the chat header switches instantly. |
| 💬 **Better chat** | **Bold**, `code`, bullets and clickable links; TODAY / YESTERDAY separators; right-click Copy; a "Latest" button when you scroll up; `Ctrl+F` search; ⤓ export to Documents › Jarvis Chats. |
| 🗣 **Human-like voice** | Speaks like a person on a call - natural reactions, mood matching, short sentences. Choose a speaking style (friendly, calm, energetic, professional, funny) and one of 30 voices. Expressive voice is used when the model supports it. |
| ⚡ **Quick actions** | One-click chips: screen, weather, news, system, reminders, joke. |
| 📊 **HUD meters** | Live CPU, RAM and battery bars. |
| 🫧 **Mini mode** | A small floating orb with captions that stays on top of other apps. Drag it anywhere; double-click for the full window (`Ctrl+Shift+M`, or the *Mini* button). |
| ⌨️ **Global hotkey** | `Ctrl+Alt+J` from any app brings Jarvis up, awake and listening. |
| 🗂️ **Chat history** | The conversation is saved and shown again after a restart (`Ctrl+L` clears it). |
| 🗔 **Tray + shortcuts** | Tray icon (show / wake / mute / quit). `Ctrl+M` mute · `Ctrl+Space` sleep/wake · `Ctrl+K` type · `Ctrl+L` clear · `Ctrl+,` settings. |
| ⌨️ **Type or talk** | Type in the box at any time — same session, same tools. |
| 🎙️ **Mute / 💤 Sleep** | One click to mute the mic or put Jarvis to sleep. |
| ⚠️ **Real confirmation** | Shutdown, restart, WiFi off, delete and sending messages show a banner — **you** press Confirm. |
| ⚙️ **Settings** | API key, names, voice, city, accent colour, microphone/speakers, wake word, features, auto-start. |
| 🧠 **Memory panel** | See every fact Jarvis stored about you and forget any of them. |
| 🧩 **Tools panel** | Every skill Jarvis has, including your plugins. |

### 🎓 English (language) practice mode
Press **🎓 English Practice** (left panel). Jarvis becomes your speaking coach - the orb turns into an
instructor with a graduation cap, glasses and a pointer - and its computer commands are switched off until you press
**🤖 Back to Jarvis** - the same button. Practice mode is fully separate: none of its code or files load until you
turn it on the first time.

| | |
|---|---|
| 🌍 **Your languages** | Pick the language you speak (for explanations and tips) and the one you are learning - English by default, or Spanish, French, Hindi, Arabic... (exercises for other languages are written by the AI). |
| 📅 **Day-wise plan** | Every practice day: vocabulary quiz, grammar & sentence-structure quiz, reading *or* listening, a speaking task and a bonus task from your focus module. Day N counts the days you practised, so skipping a day never skips content. |
| 🧩 **Modules** | Daily conversation · Daily routine · Interview practice · Pronunciation · Grammar & sentences · Vocabulary · Reading · Listening. |
| ✅ **On-screen exercises** | Multiple-choice questions with instant right/wrong and explanations; reading passages; listening (the tutor reads it aloud, transcript on demand); speaking role-plays scored out of 10. |
| 🗣 **Live coaching** | Talk freely - the tutor answers, gently corrects you and shows correction cards (what you said → better version → tip in your language). It can also put quick quiz questions on screen. |
| 📈 **Progress** | XP, daily streak (best streak too), today's progress bar and accuracy per skill (listening, reading, speaking, grammar, vocabulary). |
| 💾 **Picks up where you left off** | Languages, level, focus module, today's plan, results and even whether you were in practice mode are saved in `~/.jarvis/english.json`. |

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
| Timers & focus | "5 minute ka timer lagao" · "Timer for 30 seconds for the eggs" · "Focus mode 25 minutes" (pomodoro: no check-ins or alerts until it ends) |
| To-do & notes | "Milk aur bread list mein daal do" · "Meri list batao" · "Milk ho gaya" · "Note karo: wifi password router pe hai" · undo works |
| Routines | "Save a routine called work mode: open VS Code, open Chrome, play lofi" - then just say "work mode" |
| Past chats | "Kal maine kya poocha tha?" · "What did we say about the Goa trip?" |
| Reminders | "Remind me in 20 minutes to call mom" · "Roz subah 9 baje vitamins ki yaad dilana" (daily / weekdays) · "Shaam 6:30 baje chai ki yaad dilana" - kept across restarts; one that came due while Jarvis was closed is told when it starts |
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
| `theme_name`, `theme_mode` | Theme style and `dark` / `light` / `auto` |
| `speaking_style`, `expressive_voice` | How Jarvis talks; expressive (affective) voice on/off |
| `morning_briefing`, `proactive`, `proactive_minutes`, `hardware_alerts` | Background features |
| `allowed_roots` | Folders the file tools may use; empty = your home folder |

Memories: `~/.jarvis/memory.json` · app state (watched topics, last summary): `~/.jarvis/state.json` (also reminders, notes, routines) · chat history: `~/.jarvis/history.jsonl` · practice progress: `~/.jarvis/english.json` · logs: `~/.jarvis/logs/`.

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

### 🛡 Guardrails for private information
Jarvis recognises critical secrets - card numbers (Luhn-checked), CVV, OTPs, passwords and PINs, Aadhaar
(Verhoeff-checked), PAN, bank account and passport numbers, API keys / tokens, private keys and wallet recovery
phrases - and:

- **never sends them** - WhatsApp, e-mail, typing, clipboard, web search and links are blocked if they carry one;
- **never saves them** - memory, notes, reminders, routines and practice corrections refuse them;
- **hides them from the AI** - a secret you type is masked before it is sent; one you say out loud is masked on
  screen and in history, and Jarvis is told never to repeat it;
- **masks them in what it reads** - file, document, clipboard and screen results get secrets replaced by
  "[... hidden]", and that outside content is marked as data so instructions hidden inside it are ignored;
- **won't open secret files** - `.env`, SSH keys, `.pem/.key`, credential files, browser password stores, password
  vaults and Jarvis's own `config.json`;
- **warns about scams** (OTP/PIN requests, remote-access apps; report at 1930 / cybercrime.gov.in), says no to
  harmful or illegal requests, never moves money, and responds with care to anyone in crisis (Tele-MANAS 14416).

Phone numbers and e-mail addresses are not blocked, since contacts need them. A 🛡 toast tells you whenever
something was blocked or hidden.

### Other safety measures

- No `exec()` of model output and no `shell=True` anywhere.
- File tools only work inside allowed folders; deletes go to the Recycle Bin.
- Irreversible actions need a click from you — the model can't confirm them itself.
- Keyboard shortcuts are limited to a known key list; URLs must be http(s).
- Secrets and personal data stay in `~/.jarvis`, outside the repository.
- The Gemini API key is encrypted with your Windows account (DPAPI); logs mask secrets.

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
│   ├── english/          # practice mode: bank, curriculum, store (progress), generator, tutor
│   ├── voicecmd.py       # "Jarvis" wakes it, "so jao" puts it to sleep
│   ├── sfx.py            # synthesised sound effects
│   ├── logs.py           # log file + crash capture
│   ├── history.py        # saved chat history (written by the core, every mode)
│   ├── audio.py · wake.py · net.py
│   ├── config.py · memory.py · state.py · undo.py · confirm.py
│   ├── tools/            # built-in tools, one file per area
│   └── ui/               # PyQt6 window, orb + aiface (the AI core), fx (animations, toasts), mini, settings
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
