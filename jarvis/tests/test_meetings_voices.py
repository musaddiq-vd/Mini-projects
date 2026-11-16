"""Meeting assistant and voice profiles - with fake audio arrays and a fake model (nothing is really recorded)."""

import json
import sys
import time

import numpy as np
import pytest
from fake_voice import voice

from jarvis import meetings, permissions, voices
from jarvis.events import Events

RATE = 16_000
NOTES = {
    "summary": "The team planned the launch.",
    "decisions": ["Launch on Monday"],
    "action_items": [{"task": "Send the slides", "owner": "Priya", "due": "2026-11-20"},
                     {"task": "Book the room", "owner": "", "due": "next week"}],
    "open_questions": ["Who pays for snacks?"],
    "email": {"subject": "Launch follow-up", "body": "Hi all,\nWe launch on Monday."},
}


class FakeLLM:
    def __init__(self, notes=None):
        self.transcribed = []
        self.asked = []
        self.notes = NOTES if notes is None else notes

    def transcribe(self, audio, mime="audio/ogg", prompt=""):
        self.transcribed.append((len(audio), mime, prompt))
        return f"Speaker 1: part {len(self.transcribed)} - my OTP is 482913."

    def ask(self, prompt, system=None):
        self.asked.append(prompt)
        if "QUESTION:" in prompt:
            return "Priya will send the slides."
        return "```json\n" + json.dumps(self.notes) + "\n```"


class FakeSource:
    def __init__(self, label):
        self.label = label
        self.cb = None
        self.stopped = False

    def start(self, cb):
        self.cb = cb

    def stop(self):
        self.stopped = True


@pytest.fixture(autouse=True)
def clean_state():
    voices.set_active(None)
    meetings._current = None
    yield
    voices.set_active(None)
    if meetings._current is not None:
        meetings._current.recorder.stop(timeout=5)
    meetings._current = None


@pytest.fixture
def mctx(ctx, monkeypatch):
    ctx._llm = FakeLLM()
    ctx.events = Events()
    ctx.seen = []
    ctx.events.subscribe(lambda k, d: ctx.seen.append((k, d)))
    mic, pc = FakeSource("microphone"), FakeSource("PC sound")
    ctx.sources = (mic, pc)
    monkeypatch.setattr(meetings, "default_sources", lambda settings: ([mic, pc], ""))
    return ctx


def tone(seconds, freq=220.0, amp=0.2):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# -- recorder --------------------------------------------------------------------------------------
def test_recorder_mixes_sources_and_cuts_chunks(monkeypatch):
    monkeypatch.setattr(meetings, "MIN_TAIL_SECONDS", 0.4)
    got = []
    mic, pc = FakeSource("microphone"), FakeSource("PC sound")
    rec = meetings.Recorder([mic, pc], lambda i, s: got.append((i, s)), chunk_seconds=1.0)
    rec.start()
    pc.cb(np.full(RATE, 0.25, dtype=np.float32))
    for _ in range(5):
        mic.cb(np.full(RATE // 2, 0.1, dtype=np.float32))  # 2.5 s of microphone
    rec.stop(timeout=5)
    assert [i for i, _ in got] == [1, 2, 3]  # 1 s + 1 s + a 0.5 s tail
    first = got[0][1]
    assert len(first) == RATE and np.allclose(first, 0.35)  # mic + PC sound
    assert np.allclose(got[1][1], 0.1) and len(got[2][1]) == RATE // 2
    assert mic.stopped and pc.stopped and rec.seconds == pytest.approx(2.5)


def test_short_tail_is_dropped_and_audio_is_clipped():
    got = []
    mic = FakeSource("microphone")
    rec = meetings.Recorder([mic], lambda i, s: got.append(s), chunk_seconds=1.0)
    rec.start()
    mic.cb(np.full(RATE + RATE // 4, 3.0, dtype=np.float32))
    rec.stop(timeout=5)
    assert len(got) == 1 and got[0].max() == 1.0  # the 0.25 s tail is too short to keep


def test_to_mono_16k_and_wav_fallback():
    stereo = np.stack([tone(0.5, amp=0.4), tone(0.5, amp=0.0)], axis=1)
    mono = meetings.to_mono_16k(stereo, 48_000)
    assert mono.ndim == 1 and abs(len(mono) - RATE // 6) <= 1
    assert meetings.to_mono_16k(np.array([16384], dtype=np.int16), RATE)[0] == pytest.approx(0.5)
    data, ext, mime = meetings.encode(tone(0.2))
    assert (ext, mime) in {(".wav", "audio/wav"), (".flac", "audio/flac")} and len(data) > 1000


def test_mic_only_when_the_pc_has_no_loopback(monkeypatch):
    monkeypatch.setattr(meetings, "find_loopback", lambda: None)
    sources, note = meetings.default_sources(type("S", (), {"input_device": ""})())
    assert [s.label for s in sources] == ["microphone"] and "microphone only" in note
    monkeypatch.setattr(meetings, "find_loopback", lambda: FakeSource("PC sound"))
    sources, note = meetings.default_sources(type("S", (), {"input_device": ""})())
    assert [s.label for s in sources] == ["microphone", "PC sound"] and note == ""


# -- meeting tools ---------------------------------------------------------------------------------
def test_full_meeting_flow(mctx, registry, isolated_home):
    r = registry.call("start_meeting", {"title": "Weekly sync"}, mctx)
    assert "recording started" in r["result"].lower() and r["sources"] == ["microphone", "PC sound"]
    logs = [d["text"] for k, d in mctx.seen if k == "log" and "Recording started" in d["text"]]
    assert len(logs) == 1  # announced once
    assert [d["state"] for k, d in mctx.seen if k == "meeting"] == ["recording"]
    again = registry.call("start_meeting", {"title": "Other"}, mctx)
    assert "Already recording" in again["error"]

    mic, pc = mctx.sources
    for _ in range(int(meetings.CHUNK_SECONDS) + 30):
        mic.cb(tone(1.0))
        pc.cb(tone(1.0, 440.0, 0.1))
    status = registry.call("meeting_status", {}, mctx)
    assert status["recording"] and status["title"] == "Weekly sync" and status["minutes"] >= 1.9

    out = registry.call("stop_meeting", {}, mctx)
    card = out["meeting_notes"]
    assert card["summary"] == "The team planned the launch." and card["decisions"] == ["Launch on Monday"]
    assert card["action_items"][0] == {"task": "Send the slides", "owner": "Priya", "due": "2026-11-20"}
    assert card["action_items"][1]["due"] == ""  # not a date: dropped rather than guessed
    assert "save_meeting_actions" in out["result"] and "draft_email" in out["result"]
    assert "_safety" in out  # transcripts are outside content
    assert [d["state"] for k, d in mctx.seen if k == "meeting"] == ["recording", "processing", "stopped"]

    folder = isolated_home / "meetings" / card["id"]
    files = sorted(p.name for p in folder.iterdir())
    assert {"meeting.json", "notes.md", "transcript.txt"} <= set(files)
    assert len([f for f in files if f.startswith("chunk-")]) == 2
    text = (folder / "transcript.txt").read_text(encoding="utf-8")
    assert "482913" not in text and "part 2" in text  # secrets masked before saving
    prompts = [p for _, _, p in mctx._llm.transcribed]
    assert "Speaker 1" in prompts[0] and "part 1" in prompts[1]  # speakers kept apart across chunks
    assert "Send the slides" in (folder / "notes.md").read_text(encoding="utf-8")
    assert meetings.current() is None


def test_after_the_meeting(mctx, registry, answers):
    registry.call("start_meeting", {"title": "Budget review"}, mctx)
    mctx.sources[0].cb(tone(3.0))
    registry.call("stop_meeting", {}, mctx)

    listed = registry.call("list_meetings", {}, mctx)["meetings"]
    assert listed[0]["title"] == "Budget review" and listed[0]["notes"]
    notes = registry.call("meeting_notes", {"id": "budget"}, mctx)
    assert notes["meeting_notes"]["title"] == "Budget review"
    ans = registry.call("ask_meeting", {"id": "last", "question": "Who sends the slides?"}, mctx)
    assert ans["answer"] == "Priya will send the slides." and "Who sends the slides?" in mctx._llm.asked[-1]

    added = registry.call("save_meeting_actions", {"id": "last"}, mctx)
    assert added["added"] == ["Send the slides (Priya)", "Book the room"]
    todos = mctx.state.get("notes", []) if mctx.state else None
    from jarvis.tools.notes import _load

    items = _load(mctx) if todos is None else todos
    assert any(n.get("due") == "2026-11-20" for n in items)

    answers.append(False)
    kept = registry.call("delete_meeting", {"id": "last"}, mctx)
    assert "Cancelled" in kept["result"] and meetings.find("last") is not None
    answers.append(True)
    gone = registry.call("delete_meeting", {"id": "last"}, mctx)
    assert "Deleted" in gone["result"] and meetings.find("last") is None
    assert "couldn't find" in registry.call("meeting_notes", {"id": "last"}, mctx)["error"]


def test_meetings_permission_off_disables_recording(mctx, registry):
    permissions.set_level(mctx.settings, "meetings", "off")
    r = registry.call("start_meeting", {"title": "Secret"}, mctx)
    assert "Not allowed" in r["error"] and meetings.current() is None
    assert "Meetings" in permissions.AREAS["meetings"][0]
    assert registry.call("stop_meeting", {}, mctx)["error"] == "No meeting is being recorded."


def test_start_after_reading_outside_content_needs_a_yes(mctx, registry, answers):
    mctx.untrusted_at = time.monotonic()
    mctx.untrusted_source = "a web page"
    r = registry.call("start_meeting", {"title": "x"}, mctx)  # no "yes" given
    assert "Cancelled" in r["result"] and meetings.current() is None


def test_interrupted_meeting_still_gets_notes(mctx, registry):
    folder = meetings.meetings_dir() / "2026-11-16-0900-standup"
    folder.mkdir()
    meetings.save(folder, {"title": "Standup", "started": time.time() - 600, "status": "recording",
                           "chunks": [{"file": "chunk-001.wav", "seconds": 90, "text": "Speaker 1: hello"}]})
    out = registry.call("meeting_notes", {"id": "standup"}, mctx)
    assert out["meeting_notes"]["summary"] and meetings.load(folder)["status"] == "done"


def test_failed_chunk_is_transcribed_again_later(mctx):
    folder = meetings.meetings_dir() / "2026-11-16-1000-call"
    folder.mkdir()
    data, ext, _ = meetings.encode(tone(1.0))
    (folder / f"chunk-001{ext}").write_bytes(data)
    meetings.save(folder, {"title": "Call", "started": time.time(), "status": "processing",
                           "chunks": [{"file": f"chunk-001{ext}", "seconds": 1, "text": "", "error": "offline"}]})
    info = meetings.make_notes(mctx, folder)
    assert info["chunks"][0]["text"] and "error" not in info["chunks"][0]


def test_bad_model_answer_still_makes_a_summary(mctx, registry):
    mctx._llm.ask = lambda prompt, system=None: "Just a plain summary, no JSON."
    registry.call("start_meeting", {"title": "Chat"}, mctx)
    mctx.sources[0].cb(tone(2.0))
    out = registry.call("stop_meeting", {}, mctx)
    card = out["meeting_notes"]
    assert card["summary"] == "Just a plain summary, no JSON." and card["decisions"] == []


def test_find_and_delete_only_touch_meeting_folders(mctx, tmp_path):
    assert meetings.find("last") is None
    with pytest.raises(ValueError):
        meetings.delete(tmp_path)
    assert meetings.slug("Q4 Plan: Sales & Ops!") == "q4-plan-sales-ops"


def test_calendar_meeting_only_offers_to_take_notes(mctx):
    from jarvis.background import BackgroundServices

    fake = type("B", (), {"s": mctx.settings})()
    offer = BackgroundServices._notes_offer(fake)
    assert "Shall I take notes?" in offer and "never start recording by yourself" in offer
    mctx.settings.meeting_offer = False
    assert BackgroundServices._notes_offer(fake) == ""
    mctx.settings.meeting_offer = True
    permissions.set_level(mctx.settings, "meetings", "off")
    assert BackgroundServices._notes_offer(fake) == ""


def test_prompt_and_help_mention_both_features():
    from jarvis.app import system_prompt
    from jarvis.config import Settings
    from jarvis.memory import MemoryStore
    from jarvis.ui.help import TOPICS

    prompt = system_prompt(Settings(), MemoryStore(meetings.meetings_dir() / "m.json"))
    assert "start_meeting" in prompt and "never start it yourself" in prompt and "enroll_voice" in prompt
    titles = [t for _, t, _ in TOPICS]
    assert "Meetings" in titles and "Voice profiles" in titles


# -- voice profiles --------------------------------------------------------------------------------
SPEAKERS = {"Asha": (210, 1.18), "Ravi": (115, 1.0)}


def enroll_fake(name, f0, tract, **kw):
    return voices.enroll(name, [voice(f0, tract, 3, seed=s) for s in (1, 2, 3)], **kw)


def test_identify_tells_people_apart_and_spots_guests():
    for name, (f0, tract) in SPEAKERS.items():
        enroll_fake(name, f0, tract)
    for name, (f0, tract) in SPEAKERS.items():
        got = voices.identify(voice(f0 * 1.02, tract, 2.5, seed=11))
        assert got["name"] == name and got["sure"] and not got["guest"]
    kid = voices.identify(voice(290, 1.3, 2.5, seed=12))
    assert kid["guest"] and kid["name"] == "guest"
    silence = voices.identify(np.zeros(RATE * 2, dtype=np.int16))
    assert silence["name"] == "" and not silence["sure"]


def test_profile_keeps_only_numbers_and_first_is_owner(isolated_home):
    p = enroll_fake("Ravi", 115, 1.0, language="hi", style="calm")
    q = enroll_fake("Asha", 210, 1.18, memory="none")
    assert p["owner"] and not q["owner"] and q["memory"] == "none" and p["memory"] == "all"
    files = sorted((isolated_home / "voices").iterdir())
    assert [f.name for f in files] == ["asha.json", "ravi.json"]
    saved = json.loads(files[1].read_text(encoding="utf-8"))
    assert set(saved["voice"]) == {"mean", "scale", "phrases"} and len(saved["voice"]["mean"]) == 38
    assert files[1].stat().st_size < 6000  # no audio in there
    enroll_fake("Asha", 210, 1.18, owner=True)
    assert voices.owner_profile()["name"] == "Asha" and not voices.get_profile("Ravi")["owner"]


def test_enroll_voice_tool(ctx, registry, monkeypatch, isolated_home):
    from jarvis.tools import voice_tools

    phrases = iter([voice(115, 1.0, 4, seed=s) for s in (1, 2, 3)])
    monkeypatch.setattr(voice_tools, "record_phrase", lambda c, seconds: next(phrases))
    monkeypatch.setattr(voice_tools, "PAUSE", 0)
    ctx.events = Events()
    seen = []
    ctx.events.subscribe(lambda k, d: seen.append((k, d)))
    r = registry.call("enroll_voice", {"name": "Ravi", "language": "hi"}, ctx)
    assert r["owner"] and "not the recording" in r["result"]
    shown = [d["phrase"] for k, d in seen if k == "voice_enroll" and d["phrase"]]
    assert shown == list(voices.PHRASES)
    assert not list((isolated_home / "voices").glob("*.wav"))

    monkeypatch.setattr(voice_tools, "record_phrase", lambda c, seconds: np.zeros(RATE * 4, dtype=np.int16))
    quiet = registry.call("enroll_voice", {"name": "Asha"}, ctx)
    assert "couldn't hear" in quiet["error"] and voices.get_profile("Asha") is None


def test_profile_tools(ctx, registry, answers):
    assert "No voice profiles" in registry.call("who_is_speaking", {}, ctx)["result"]
    enroll_fake("Ravi", 115, 1.0)
    enroll_fake("Asha", 210, 1.18, style="funny")
    listed = registry.call("list_voice_profiles", {}, ctx)
    assert [p["name"] for p in listed["profiles"]] == ["Asha", "Ravi"] and listed["profiles"][0]["style"] == "funny"
    assert "haven't recognised" in registry.call("who_is_speaking", {}, ctx)["result"]
    voices.set_active(voices.get_profile("Asha"))
    assert registry.call("who_is_speaking", {}, ctx)["speaker"] == "Asha"
    answers.append(True)
    assert "Deleted" in registry.call("delete_voice_profile", {"name": "asha"}, ctx)["result"]
    assert voices.get_profile("Asha") is None and voices.active() == {}
    voices.set_active(None, guest=True)
    assert "Only a recognised voice" in registry.call("delete_voice_profile", {"name": "Ravi"}, ctx)["error"]


def test_guest_cannot_use_whatsapp_or_money_unless_allowed(ctx, registry):
    voices.set_active(None, guest=True)
    for name in ("send_whatsapp", "add_expense", "find_contact"):
        decision, text = permissions.check(name, {}, ctx.settings)
        assert decision == permissions.OFF and "guests" in text
    assert permissions.check("get_time", {}, ctx.settings)[0] == permissions.ON
    assert permissions.check("send_whatsapp", {}, ctx.settings, source="phone")[0] == permissions.ON
    r = registry.call("expense_summary", {}, ctx)
    assert "Not allowed for guests" in r["error"]
    ctx.settings.guest_private_tools = True
    assert permissions.check("add_expense", {}, ctx.settings)[0] == permissions.ON
    ctx.settings.guest_private_tools = False
    voices.set_active(None)
    assert permissions.check("add_expense", {}, ctx.settings)[0] == permissions.ON


def test_watcher_switches_only_on_confident_changes():
    from jarvis.config import Settings

    ravi = enroll_fake("Ravi", 115, 1.0)
    enroll_fake("Asha", 210, 1.18)
    changes, events = [], []
    bus = Events()
    bus.subscribe(lambda k, d: events.append(k))
    w = voices.VoiceWatcher(Settings(), bus, on_change=lambda p, g, prev: changes.append((p and p["name"], g, prev)),
                            background=False)

    def say(f0, tract, seed):
        pcm = voice(f0, tract, 2.5, seed=seed).tobytes()
        for i in range(0, len(pcm), 2560):  # 80 ms mic chunks, like live.py
            w.feed(pcm[i:i + 2560])
        w.end_turn()

    say(115, 1.0, 20)
    assert changes == [("Ravi", False, {})] and voices.active()["name"] == "Ravi" == ravi["name"]
    say(210, 1.18, 21)  # one turn by someone else is not enough
    assert len(changes) == 1
    say(210, 1.18, 24)
    assert changes[-1][:2] == ("Asha", False) and changes[-1][2]["name"] == "Ravi"
    say(290, 1.3, 23)
    say(290, 1.3, 24)
    assert changes[-1][:2] == (None, True) and voices.is_guest()
    assert "speaker" in events and "voice_profile" in events

    w.s.voice_id = False
    w.feed(voice(115, 1.0, 2.5, seed=25).tobytes())
    w.end_turn()
    assert voices.is_guest()  # recognition off: nothing changes


def test_watcher_never_blocks_and_skips_while_busy():
    from jarvis.config import Settings

    enroll_fake("Ravi", 115, 1.0)
    calls = []
    w = voices.VoiceWatcher(Settings(), identify_fn=lambda a, p: calls.append(len(a)) or {"name": ""},
                            background=False)
    w._busy.acquire()  # a check is still running
    w.feed(voice(115, 1.0, 2, seed=3))
    w.end_turn()
    assert calls == [] and w._len == 0
    w._busy.release()
    for _ in range(250):  # 20 s of 80 ms chunks
        w.feed(np.zeros(1280, dtype=np.int16))
    assert w._len <= voices.MAX_TURN_SECONDS * RATE + RATE  # only the last seconds are kept
    w.end_turn()
    assert len(calls) == 1


def test_system_notes():
    from jarvis.config import Settings

    owner = enroll_fake("Ravi", 115, 1.0)
    asha = enroll_fake("Asha", 210, 1.18, language="hi", style="calm", memory="own")
    note = voices.system_note(asha, False, Settings())
    assert "Asha is speaking" in note and "Hindi" in note and "calm" in note and "'asha'" in note
    guest = voices.system_note(None, True, Settings())
    assert "guest" in guest and "Ravi" in guest and "WhatsApp" in guest
    assert "owner" in voices.system_note(owner, False, Settings())
    voices.set_active(asha)
    assert "Asha" in voices.prompt_line()


def test_live_turns_feed_the_voice_watcher(isolated_home):
    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.live import LiveAssistant

    a = LiveAssistant(create_app(Settings(api_key="k", sound_effects=False)))
    calls = []
    a.voice_watch = type("W", (), {"end_turn": lambda self: calls.append("end"),
                                   "drop": lambda self: calls.append("drop")})()
    a._user_turn(["hello there"])
    a._user_turn([])
    assert calls == ["end", "drop"]
    sent = []
    a.send_system = sent.append
    owner = enroll_fake("Ravi", 115, 1.0)
    a._voice_changed(owner, False, {})  # the owner at the start: nothing to say
    assert sent == []
    a._voice_changed(None, True, {"name": "Ravi"})
    assert sent and sent[0].startswith("(system: voice)")


# -- window ----------------------------------------------------------------------------------------
@pytest.fixture
def qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


def test_meeting_card_and_pill(qt):
    from PyQt6.QtWidgets import QLabel

    from jarvis.ui.cards import meeting_card
    from jarvis.ui.meeting_pill import MeetingPill, rec_text

    assert meeting_card({"result": "x"}) is None
    card = meeting_card({"meeting_notes": {"title": "Sync <b>", "when": "Mon", "sources": ["microphone"],
                                           **meetings.clean_notes(NOTES)}})
    text = " ".join(lb.text() for lb in card.findChildren(QLabel))
    assert "Send the slides" in text and "Priya" in text and "Launch follow-up" in text and "&lt;b&gt;" in text

    pill = MeetingPill()
    assert pill.isHidden()
    pill.on_event({"state": "recording", "title": "Weekly sync", "started": time.time() - 65})
    assert not pill.isHidden() and pill.text().startswith("● REC 1:05")
    pill.on_event({"state": "processing"})
    assert "notes" in pill.text()
    pill.on_event({"state": "stopped"})
    assert pill.isHidden()
    assert rec_text("A very long meeting title indeed", 3725).startswith("● REC 1:02:05")


def test_privacy_lists_and_deletes_profiles_and_meetings(qt, mctx, registry, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.privacy import PrivacyTab

    enroll_fake("Ravi", 115, 1.0)
    registry.call("start_meeting", {"title": "Retro"}, mctx)
    mctx.sources[0].cb(tone(2.0))
    registry.call("stop_meeting", {}, mctx)
    app = create_app(Settings())
    tab = PrivacyTab(app, app.settings)
    box = tab.voices_meetings
    labels = " ".join(w.text() for w in box.findChildren(type(tab.backup_info)))
    assert "Ravi" in labels and "Retro" in labels and "owner" in labels
    box.guest_tools.setChecked(True)
    box.voice_id.setChecked(False)
    s = Settings()
    tab.apply_to(s)
    assert s.guest_private_tools and not s.voice_id and s.meeting_offer
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    box._delete_voice("Ravi")
    assert voices.load_profiles() == []
    box.delete_all()
    assert meetings.all_meetings() == []
