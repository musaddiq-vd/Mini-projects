"""Settings glue of the main window: the settings dialog, phone pairing, practice mode and reminder buttons."""

from __future__ import annotations

import logging

from jarvis.config import save_settings
from jarvis.ui.settings_dialog import SettingsDialog, apply_settings

log = logging.getLogger("jarvis.ui")

RESTART_KEYS = {"voice", "speaking_style", "expressive_voice", "live_model", "input_device", "output_device", "wake_word",
                "assistant_name", "user_name", "api_key", "allow_interrupt"}


class SettingsMixin:
    """Applies changed settings and switches practice mode."""

    # -- practice mode -------------------------------------------------------------------
    practicing = False

    def set_practice(self, on: bool) -> None:
        """English practice mode: the tutor takes over; Jarvis's commands are off until switched back."""
        if on == self.practicing:
            return
        self.practicing = on
        self.app.state.set("practice_mode", on)
        if on and self.practice is None:
            from jarvis.ui.practice import PracticePanel  # first use: only now does practice code load

            self.practice = PracticePanel(self.app, None)
            self.stack.addWidget(self.practice)
        if self.practice is not None:
            self.app.english.set_mode("practice" if on else "agent")
        if self.assistant is not None:
            self.assistant.set_mode("practice" if on else "agent")
            if on and not self.assistant.awake:
                self.assistant.wake_up()  # the user wants to practise now
        if self.practice is not None:
            # announce(): the tutor is heard even if Jarvis dozes off between tasks
            self.practice.say = self.assistant.announce if self.assistant is not None else None
        self.stack.setCurrentWidget(self.practice if on else self.chat)
        self.orb.set_persona("tutor" if on else "")
        self.chips.setVisible(not on)
        self.header_label.setText("LANGUAGE PRACTICE" if on else "CONVERSATION")
        if on:
            self.close_search()
        target = self.app.english.profile.get("target", "English")
        self.app.state.set("practice_target", target)
        if on:
            self.practice_btn.setText("🤖  Back to Jarvis (agent mode)")
            self.practice_btn.setObjectName("modeOn")
            self.input.setPlaceholderText(f"Talk or type in {target} - your tutor will help...")
            self.title.setText(f"{self.app.settings.assistant_name.upper()}  ·  TUTOR")
            self.practice.reload()
            self.toast("🎓", f"{target} practice mode - commands are off until you press Back to Jarvis.",
                       self.accent, 4)
        else:
            self.practice_btn.setText(f"🎓  {target} Practice")
            self.practice_btn.setObjectName("primary")
            self.input.setPlaceholderText("Type a message, or just talk...  (Ctrl+K)")
            self.title.setText(self.app.settings.assistant_name.upper())
            self.toast("🤖", "Back to Jarvis - commands are on again.", self.accent, 3)
        self.practice_btn.style().unpolish(self.practice_btn)
        self.practice_btn.style().polish(self.practice_btn)

    phone = None

    def apply_phone(self) -> None:
        """Run the Telegram link when a paired bot is set up (restarted after a change)."""
        s = self.app.settings
        if self.phone is not None:
            self.phone.stop()
            self.phone = None
        if s.telegram_token and s.telegram_chat_id:
            from jarvis.phone import PhoneBridge

            self.phone = PhoneBridge(self.app)
            self.phone.start()

    def phone_pair(self, token: str) -> tuple[str, str]:
        """Settings > Phone > Connect: check the token and start listening for the pairing code."""
        from jarvis.phone import PhoneBridge, TelegramAPI

        s = self.app.settings
        bridge = PhoneBridge(self.app, api=TelegramAPI(token))
        bot = bridge.check_token()  # raises a friendly error for a wrong token
        if self.phone is not None:
            self.phone.stop()
        if s.telegram_token != token:
            s.telegram_token, s.telegram_chat_id, s.telegram_user = token, 0, ""
        code = bridge.begin_pairing()
        self.phone = bridge
        bridge.start()
        return bot, code

    def _language_notice(self) -> str:
        from jarvis.lang import prompt_rules

        return ("(system: the user just changed the language settings; this replaces earlier language rules) "
                + prompt_rules(self.app.settings).replace("\n", " ") + " Reply only with a short confirmation.")

    def _reminder_action(self, name: str, args: dict) -> None:
        """A button on a reminder pop-up: run the same tool the voice command uses (off the GUI thread)."""
        import threading

        def run():
            try:
                self.app.registry.call(name, args, self.app.ctx)
            except Exception:
                log.exception("reminder action %s failed", name)

        threading.Thread(target=run, daemon=True).start()

    def open_settings(self, first_run: bool = False) -> bool:
        dialog = SettingsDialog(self.app, self, first_run=first_run is True)
        if not dialog.exec():
            return False
        changed = apply_settings(self.app.settings, dialog.result_settings)
        save_settings(self.app.settings)
        if changed & {"theme_color", "theme_name", "theme_mode", "assistant_name", "orb_face", "avatar",
                      "theme_second"}:
            self._apply_theme()
        if changed & {"telegram_token", "telegram_chat_id"}:
            self.apply_phone()
        if changed & {"desktop_widget", "widget_corner", "widget_screen", "theme_color", "theme_name"}:
            self.apply_desktop_widget()
        if "history_days" in changed and self.app.history:
            self.app.history.prune(self.app.settings.history_days)
        if self.assistant is not None and "power_up_wake" in changed:
            self.assistant.sfx.power_up = self.app.settings.power_up_wake
        if self.assistant is not None and "sound_style" in changed:
            from jarvis.sfx import build_sounds

            self.assistant.sfx.style = self.app.settings.sound_style
            self.assistant.sfx.sounds = build_sounds(self.app.settings.sound_style)
            self.assistant.sfx.play("action")  # let the user hear the new style
        if self.assistant is not None and changed & {"sound_effects", "sound_volume"}:
            self.assistant.sfx.enabled = self.app.settings.sound_effects
            self.assistant.sfx.volume = self.app.settings.sound_volume
            self.assistant.sfx.play("success")
        if self.assistant is not None and "voice_effect" in changed and self.assistant.speaker is not None:
            from jarvis import voicefx

            self.assistant.speaker.fx = voicefx.make(self.app.settings.voice_effect)  # takes effect at once
        if self.assistant is None:
            self.start_assistant()
        elif changed & RESTART_KEYS:
            self.assistant.restart(fresh="voice" in changed)
        elif changed & {"reply_language", "reminder_language"}:
            self.assistant.send_system(self._language_notice())  # no restart: the chat goes on
        if changed:
            self._add_system("Settings saved.")
        return True
