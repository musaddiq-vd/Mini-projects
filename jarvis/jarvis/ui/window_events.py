"""Handles events from the Jarvis core: state, captions, tool chips and cards, toasts and emotes."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QTimer

from jarvis.ui import theme
from jarvis.ui.chat_view import pretty_tool, tool_icon
from jarvis.ui.theme import DANGER, OK, STATE_LABELS, WARN

log = logging.getLogger("jarvis.ui")

THANKS_WORDS = ("thank", "thanks", "shukriya", "dhanyavad", "dhanyawad", "love you", "awesome", "great job",
                "well done", "you're the best", "nice work", "शुक्रिया", "धन्यवाद", "شکریہ")
PRAISE_WORDS = ("wow", "awesome", "amazing", "great", "nice", "cool", "perfect", "superb", "mast", "zabardast",
                "badhiya", "kamaal", "shandaar", "बढ़िया", "ज़बरदस्त", "कमाल")
CELEBRATE_WORDS = ("congratulations", "congrats", "mubarak", "well done", "great job", "shabash", "proud of you",
                   "bahut badhiya", "excellent", "बधाई", "मुबारक", "شاباش", "مبارک")
JOKE_WORDS = ("joke", "mazak", "mazaak", "chutkula", "funny", "मज़ाक", "चुटकुला")
AWAKE_STATES = {"listening", "speaking", "muted", "thinking"}


class EventsMixin:
    """The event dispatcher and what each event shows."""

    def _planner_follow(self, kind: str) -> None:
        """Keep an open planner in sync when Jarvis changes lists by voice."""
        planner = getattr(self, "planner", None)
        if planner is not None and planner.isVisible() and kind in ("tool", "timer", "timer_end"):
            QTimer.singleShot(0, planner.refresh)  # after this event has been handled
        dashboard = getattr(self, "dashboard", None)
        if dashboard is not None and dashboard.isVisible() and kind == "tool":
            QTimer.singleShot(0, dashboard.refresh)

    def _forward(self, kind: str, data: dict) -> None:
        """Worker thread -> Qt thread. After the window is gone, events are dropped quietly."""
        try:
            self.bridge.event.emit(kind, data)
        except RuntimeError:
            pass

    # -- events ---------------------------------------------------------------------
    def _on_event(self, kind: str, data: dict) -> None:
        self._planner_follow(kind)
        if kind == "level":
            self.orb.set_level(data["value"])
        elif kind == "state":
            self._on_state(data["state"])
        elif kind == "user":
            self._add_bubble("You", data["text"], mine=True)
            if self.mini:
                self.mini.add_message("user", data["text"])
            self._react_to_user(data["text"])
        elif kind == "assistant":
            self._add_bubble(self.app.settings.assistant_name, data["text"], typewriter=True)
            if self.mini:
                self.mini.add_message("assistant", data["text"])
            if self._joke_pending:
                self._joke_pending = False
                self._emote("wink")
            elif any(w in data["text"].lower() for w in CELEBRATE_WORDS):
                self._emote("star")
        elif kind == "reminder_set":
            self.reminders.show("set", data["id"], data.get("message", ""), data.get("due", 0),
                                repeat=data.get("repeat", ""))
        elif kind == "reminder_due":
            self.reminders.show("due", data["id"], data.get("message", ""), data.get("due", 0),
                                late=data.get("late", False), repeat=data.get("repeat", ""))
            self.orb.flash()
        elif kind == "language":
            from jarvis.lang import label

            self.toast("🌐", f"Jarvis: {label(data.get('reply', 'auto'))}  ·  Reminders: "
                       f"{label(data.get('reminders', 'same'))}", theme.ACCENT, 4)
        elif kind == "motion" and data.get("state") == "running":
            self.toast("🖱", "Scrolling… say “ruko” or “stop” when it's enough", self.accent, 4)
        elif kind == "offline_mode" and data.get("on"):
            self.toast("📴", "No internet - timers, reminders, notes, your list and apps still work. Type, or "
                       "say “Jarvis, …” in English.", WARN, 8)
        elif kind == "phone":
            who = data.get("who")
            text = data.get("text", "")
            if who == "user":
                self._add_bubble("📱 You", text, mine=True)
            else:
                self._add_bubble(f"{self.app.settings.assistant_name} → 📱", text)
        elif kind == "phone_paired" and data.get("who"):
            self.toast("📱", f"Phone connected: {data['who']}", OK, 5)
        elif kind == "restart_requested":
            self.restart_app()
        elif kind == "reminder_closed":
            self.reminders.close(data["id"])
        elif kind == "document":
            name = data.get("name", "")
            self.doc_chip.setText(f"📄  {name.replace('&', '&&')}  ·  ask anything about it   ✕")
            self.doc_chip.setVisible(bool(name))
        elif kind == "nudge":
            icon = {"eyes": "👀", "water": "💧", "recap": "🌙", "meeting": "📅", "weekly": "📊", "todo": "📝"}.get(
                data.get("kind"), "💡")
            text = data.get("text", "")
            self.toast(icon, text[1:].strip() if text[:1] == icon[:1] else text, theme.ACCENT,
                       15 if data.get("kind") == "meeting" else 8)
            if self.tray and not self.isActiveWindow():
                self.tray.showMessage(self.app.settings.assistant_name, data.get("text", ""))
            if data.get("kind") in ("eyes", "water"):
                self.orb.transform("happy", 2.0)
        elif kind == "ui_command":
            self.ui_command(data.get("action", ""), data.get("theme", ""), data)
        elif kind == "achievement":
            from jarvis.ui.fx import Confetti

            Confetti(self.centralWidget())
            self.orb.transform("star", 3.2)
            self.toast(data.get("icon", "🏆"), f"Achievement unlocked: {data.get('title')} - {data.get('desc')}",
                       "#ffd23f", 6)
            self._add_system(f"🏆 Achievement unlocked: {data.get('icon', '')} {data.get('title')}")
        elif kind == "capture":
            if self.app.settings.notify_capture:
                self.flash_capture(data.get("what", "screen"))
        elif kind == "guardrail":
            if data.get("action") == "blocked":
                self.toast("🛡", "Blocked for your safety - Jarvis never sends or saves passwords, OTPs, "
                                 "card or ID numbers.", DANGER, 6)
                self._add_system("🛡 A risky action was blocked to protect your private information.")
            else:
                self.toast("🛡", f"Your {data.get('text', 'private info')} was hidden for your safety.", WARN, 5)
            self.orb.emote("surprised", 1.6)
        elif kind == "practice_answer":
            if data.get("correct"):
                self.orb.transform("happy", 1.3)
            else:
                self.orb.emote("sad", 1.6)
        elif kind.startswith("practice_") and self.practice is None:
            pass  # practice events only matter once the practice screen exists
        elif kind == "practice_mcq":
            self.practice.tutor_mcq(data["question"], data["options"], data["answer"], data.get("why", ""))
        elif kind == "practice_correction":
            self.practice.correction(data.get("said", ""), data["better"], data.get("tip", ""))
            self.orb.point()
        elif kind == "practice_progress":
            self.practice.progress(**data)
            share = data.get("score", 0) / (data.get("total") or 1)
            if share >= 0.9:
                self._emote("star", 3.2)
            elif share >= 0.6:
                self._emote("happy")
            if data.get("xp_gained"):
                self.toast("⭐", f"+{data['xp_gained']} XP  ·  🔥 {data.get('streak', 0)} day streak", OK, 3)
        elif kind == "timer":
            self.orb.set_timer(data["id"], data["ends"], data["total"], data.get("label", ""))
        elif kind == "timer_end":
            self.orb.end_timer(data["id"])
            if data.get("done"):
                self._emote("surprised")
                self.orb.flash()
                self.toast("⏱", f"{data.get('label') or 'Timer'} - time's up!", OK, 8)
                if self.tray and not self.isActiveWindow():
                    self.tray.showMessage(self.app.settings.assistant_name,
                                          f"⏱ {data.get('label') or 'Timer'} - time's up!")
        elif kind == "caption":
            self._on_caption(data.get("who", ""), data.get("text", ""))
        elif kind == "tool_start":
            name = data["name"]
            self.orb.action(f"{tool_icon(name)}  {pretty_tool(name)}...")
        elif kind == "tool":
            result = data.get("result", {})
            ok = "error" not in result
            self.orb.pulse(ok)
            if not ok:
                self._emote("sad")
            elif data["name"] == "complete_note":
                self._emote("wink")
            self.orb.action(f"{'✓' if ok else '✗'}  {pretty_tool(data['name'])}", OK if ok else DANGER)
            card = self._rich_card(data["name"], result) if ok else None
            if card is not None:  # a card says more than the plain action chip
                self.chat._add_row(card, "left", text=f"{data['name']} {result.get('place', '')} {result.get('query', '')}")
            else:
                self.chat.add_tool(data["name"], data.get("args", {}), result)
            image = result.get("image") if ok else None
            if image and Path(image).exists():
                caption = Path(image).name if "Screenshot" in Path(image).name else "What I saw"
                self.chat.add_image(image, f"🖼  {caption}  ·  click to open", self.accent)
        elif kind == "agents":
            self._on_plan(data)
        elif kind == "crash":
            self._add_system(f"⚠ Something went wrong: {data['text']}  (details in the log: tray → Open logs folder)")
            self.toast("⚠", "Something went wrong - details are in the log.", DANGER)
        elif kind == "log":
            text = data["text"]
            self._add_system(text)
            if text.startswith("🔔"):
                self.toast("🔔", text[1:].strip(), WARN, 8)
            elif text.startswith("✓ Reconnected"):
                self.toast("📶", "Back online.", OK)
            elif text.startswith("[connection lost]") and not self._offline_toast:
                self._set_net("reconnecting")
                self._offline_toast = True
                self.toast("📡", "Connection lost - reconnecting...", WARN)
            if text.startswith("✓ Reconnected"):
                self._offline_toast = False
            if text.startswith("🔔") and self.tray and not self.isActiveWindow():
                self.tray.showMessage(self.app.settings.assistant_name, text[1:].strip())

    def _on_plan(self, data: dict) -> None:
        """The agent team's / a workflow's live plan card: drawn once per run, then updated in place."""
        from jarvis.ui.cards import plan_card

        cards = self.__dict__.setdefault("_plan_cards", {})
        card = cards.get(data.get("run_id"))
        try:
            if card is not None:
                card.update_plan(data)
                return
            card = plan_card(data)
        except RuntimeError:  # the card was cleared from the chat
            card = plan_card(data)
        except Exception:
            log.exception("couldn't draw the plan card")
            return
        if card is not None:
            cards.clear()  # only the newest run is updated live
            cards[data.get("run_id")] = card
            self.chat._add_row(card, "left", text=f"plan {data.get('goal', '')}")

    def _rich_card(self, name: str, result: dict):
        from jarvis.ui.cards import digest_card, fix_card, links_card, plan_card, report_card, weather_card

        try:
            if name == "browse_and_report":
                return links_card(result)
            if name == "agent_status":
                return plan_card(result)
            if name == "get_weather":
                return weather_card(result)
            if name == "web_search":
                return links_card(result)
            if name == "weekly_report":
                return report_card(result)
            if name == "fix_on_screen":
                return fix_card(result)
            if name == "daily_digest":
                return digest_card(result)
        except Exception:
            log.exception("couldn't draw a card for %s", name)
        return None

    _joke_pending = False
    _offline_toast = False

    def _emote(self, kind: str, seconds: float = 2.8) -> None:
        """Happy feelings morph the core into the cute face for a moment; others show on the AI core."""
        if kind in ("love", "happy", "wink", "star"):
            self.orb.transform(kind, seconds)
        else:
            self.orb.emote(kind)
        self.app.events.emit("emote", feeling=kind)  # sound effects listen for this

    def _react_to_user(self, text: str) -> None:
        low = text.lower()
        if any(w in low for w in THANKS_WORDS):
            self._emote("love")
        elif any(w in low for w in PRAISE_WORDS):
            self._emote("happy")
        if any(w in low for w in JOKE_WORDS):
            self._joke_pending = True

    def _set_net(self, status: str) -> None:
        """Small pill under the clock: Online / Connecting / Reconnecting / Offline."""
        text, color = {
            "online": ("● Online", OK), "connecting": ("● Connecting…", WARN),
            "reconnecting": ("● Reconnecting…", WARN), "offline": ("● Offline", DANGER),
        }.get(status, ("● Offline", DANGER))
        if status == "online":
            text += f"  ·  {self.app.settings.live_model}"
        self.net_status = status
        self.net_pill.setText(text)
        self.net_pill.setStyleSheet(f"color: {color}; border: 1px solid {color}; border-radius: 9px; "
                                    "padding: 1px 10px; font-size: 10px; font-weight: 600; background: transparent;")

    def _on_state(self, state: str) -> None:
        prev, self._last_state = self._last_state, state
        if state == "connecting":
            self._set_net("reconnecting" if getattr(self, "net_status", "") == "online" else "connecting")
        elif state == "offline":
            self._set_net("offline")
        elif state != "thinking":
            self._set_net("online")
        self.orb.set_state(state)
        self.state_label.setText(STATE_LABELS.get(state, state.upper()))
        self._set_state_style(state)
        self.sleep_btn.setText("☀  Wake" if state == "sleeping" else "💤  Sleep")
        name = self.app.settings.assistant_name
        if state == "thinking":
            self.typing.show_text(f"{name} is working on it...", WARN)
        elif state == "connecting":
            self.typing.show_text("Connecting...", theme.MUTED)
        else:
            self.typing.hide_now()
        if prev == "sleeping" and state in AWAKE_STATES:
            if self.app.settings.power_up_wake:
                self.orb.power_up()
            else:
                self.orb.flash()
        if state == "sleeping":
            how = self.assistant.how_to_wake() if self.assistant else "press Wake"
            mode = self.assistant.wake_mode if self.assistant else ""
            # privacy at a glance: is anything still being heard while asleep?
            where = {"wake_word": "🔒 offline wake word - nothing leaves your PC",
                     "name": "☁ listening for your name online"}.get(mode, "🔇 mic off until you press Wake")
            self.hint.setText(f"{how[0].upper() + how[1:]}  ·  {where}")
            self.caption.setText("")
        else:
            self.hint.setText("")

    def _on_caption(self, who: str, text: str) -> None:
        if who == "user" and text and self._last_state != "thinking":
            self.typing.show_text(f"{self.app.settings.assistant_name} is listening...", self.accent)
        if not text:
            QTimer.singleShot(1800, self._clear_caption)
            self._caption_live = False
            return
        self._caption_live = True
        if len(text) > 140:
            text = "…" + text[-140:]
        color = (theme.readable(OK) if who == "assistant" else theme.TEXT)
        icon = "🔊" if who == "assistant" else "🎙"
        self.caption.setStyleSheet(f"color:{color};")
        self.caption.setText(f"{icon}  {text}")
        if self.mini:
            self.mini.set_caption(f"{icon} {text}")

    def _clear_caption(self) -> None:
        if self._caption_live:
            return
        self.caption.setText("")
        if self.mini:
            self.mini.set_caption("")

    _caption_live = False
