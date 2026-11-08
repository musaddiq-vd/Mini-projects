"""Settings window: account, look, audio, wake word, features, memory and tools."""

from __future__ import annotations

import copy
import platform
from datetime import datetime

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from jarvis.app import Jarvis
from jarvis.config import SPEAKING_STYLES, VOICE_INFO, Settings
from jarvis.ui.theme import MODES, THEMES, valid_color

WAKE_WORDS = ["", "hey_jarvis", "alexa", "hey_mycroft", "hey_rhasspy"]
PRESET_COLORS = ["#00d4ff", "#00ff9c", "#ff3df2", "#ffb547", "#ff5c6c", "#8b7bff", "#ffffff"]


class SettingsDialog(QDialog):
    def __init__(self, app: Jarvis, parent=None, first_run: bool = False):
        super().__init__(parent)
        self.app = app
        self.result_settings: Settings = copy.deepcopy(app.settings)
        self.setWindowTitle("Settings")
        self.setMinimumSize(760, 580)

        layout = QVBoxLayout(self)
        if first_run:
            hello = QLabel("Welcome! Paste your free Gemini API key to get started "
                           "(get one at aistudio.google.com/apikey).")
            hello.setWordWrap(True)
            layout.addWidget(hello)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.tabs.addTab(self._general_tab(), "General")
        self.tabs.addTab(self._look_tab(), "Look")
        self.tabs.addTab(self._audio_tab(), "Audio")
        self.tabs.addTab(self._features_tab(), "Features")
        self.tabs.addTab(self._memory_tab(), "Memory")
        self.tabs.addTab(self._tools_tab(), "Tools")
        from jarvis.ui.privacy import PrivacyTab

        from PyQt6.QtWidgets import QScrollArea

        self.privacy = PrivacyTab(app, self.result_settings)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.privacy)
        self.tabs.addTab(scroll, "🔒 Privacy")

        buttons = QHBoxLayout()
        buttons.addWidget(QLabel("Jarvis · by Musaddiq Khan", objectName="muted"))
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save", objectName="primary")
        save.clicked.connect(self._save)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        layout.addLayout(buttons)

    # -- tabs -----------------------------------------------------------------
    def _general_tab(self) -> QWidget:
        s = self.result_settings
        w = QWidget()
        form = QFormLayout(w)
        self.api_key = QLineEdit(s.api_key)
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("Gemini API key")
        show = QPushButton("Show")
        show.setCheckable(True)
        show.toggled.connect(lambda on: self.api_key.setEchoMode(
            QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        row = QHBoxLayout()
        row.addWidget(self.api_key)
        row.addWidget(show)
        form.addRow("API key", row)
        self.user_name = QLineEdit(s.user_name)
        form.addRow("Your name", self.user_name)
        self.assistant_name = QLineEdit(s.assistant_name)
        form.addRow("Assistant name", self.assistant_name)
        self.voice = QComboBox()
        for name, sound in VOICE_INFO.items():
            self.voice.addItem(f"{name}  -  {sound}", name)
        self.voice.setCurrentIndex(max(0, self.voice.findData(s.voice)))
        form.addRow("Voice", self.voice)
        self.style = QComboBox()
        for key, desc in SPEAKING_STYLES.items():
            self.style.addItem(f"{key.title()}  -  {desc.split(',')[0]}", key)
        self.style.setCurrentIndex(max(0, self.style.findData(s.speaking_style)))
        form.addRow("Speaking style", self.style)
        self.expressive = QCheckBox("Expressive voice - the tone follows the mood of the conversation")
        self.expressive.setChecked(s.expressive_voice)
        form.addRow("", self.expressive)
        from jarvis.lang import LANGUAGES, label

        self.reply_lang = QComboBox()
        self.reply_lang.addItem("🌐  " + label("auto"), "auto")
        for code in LANGUAGES:
            self.reply_lang.addItem(label(code), code)
        self.reply_lang.setCurrentIndex(max(0, self.reply_lang.findData(s.reply_language)))
        self.reply_lang.setToolTip("The language Jarvis speaks and writes in. Auto follows whatever you speak.")
        form.addRow("Jarvis speaks", self.reply_lang)
        self.reminder_lang = QComboBox()
        self.reminder_lang.addItem("↻  " + label("same"), "same")
        self.reminder_lang.addItem("🌐  " + label("auto"), "auto")
        for code in LANGUAGES:
            self.reminder_lang.addItem(label(code), code)
        self.reminder_lang.setCurrentIndex(max(0, self.reminder_lang.findData(s.reminder_language)))
        self.reminder_lang.setToolTip("The language of reminder pop-ups and of Jarvis saying them.")
        form.addRow("Reminders in", self.reminder_lang)
        self.city = QLineEdit(s.city)
        self.city.setPlaceholderText("For weather in the morning briefing")
        form.addRow("City", self.city)
        self.live_model = QLineEdit(s.live_model)
        form.addRow("Voice model", self.live_model)
        self.text_model = QLineEdit(s.text_model)
        form.addRow("Text model", self.text_model)
        return w

    def _look_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel("Theme", objectName="section"))
        form = QFormLayout()
        self.theme_name = QComboBox()
        self.theme_name.addItems(list(THEMES))
        self.theme_name.setCurrentText(self.result_settings.theme_name)
        self.theme_name.currentTextChanged.connect(lambda name: self._set_color(THEMES[name][0]))
        form.addRow("Style", self.theme_name)
        self.theme_mode = QComboBox()
        for m, label in zip(MODES, ("🌙  Night (dark)", "☀  Day (light)", "🌓  Auto - day by day, night by night")):
            self.theme_mode.addItem(label, m)
        self.theme_mode.setCurrentIndex(max(0, self.theme_mode.findData(self.result_settings.theme_mode)))
        form.addRow("Mode", self.theme_mode)
        v.addLayout(form)
        v.addSpacing(10)
        v.addWidget(QLabel("Accent colour (optional - overrides the theme's)", objectName="section"))
        row = QHBoxLayout()
        self.color = valid_color(self.result_settings.theme_color)
        self.swatches: list[QPushButton] = []
        for c in PRESET_COLORS:
            b = QPushButton()
            b.setFixedSize(34, 34)
            b.setStyleSheet(f"background:{c}; border-radius:17px; border:2px solid #222;")
            b.clicked.connect(lambda _, c=c: self._set_color(c))
            row.addWidget(b)
        pick = QPushButton("Custom...")
        pick.clicked.connect(self._pick_color)
        row.addWidget(pick)
        row.addStretch()
        v.addLayout(row)
        self.color_label = QLabel()
        v.addWidget(self.color_label)
        self._set_color(self.color)
        v.addSpacing(12)
        v.addWidget(QLabel("Orb", objectName="section"))
        self.orb_face = QCheckBox("Show the AI face on the core (LED eyes, voice waveform, feelings)")
        self.orb_face.setChecked(self.result_settings.orb_face)
        v.addWidget(self.orb_face)
        self.power_up = QCheckBox("Power-up animation and sound when Jarvis wakes up")
        self.power_up.setChecked(self.result_settings.power_up_wake)
        v.addWidget(self.power_up)
        desk_row = QHBoxLayout()
        self.desk_widget = QCheckBox("Desktop widget on the wallpaper (time, what's next, habits)")
        self.desk_widget.setChecked(self.result_settings.desktop_widget)
        desk_row.addWidget(self.desk_widget, 1)
        self.desk_corner = QComboBox()
        for key, text in (("top_right", "Top right"), ("top_left", "Top left"),
                          ("bottom_right", "Bottom right"), ("bottom_left", "Bottom left")):
            self.desk_corner.addItem(text, key)
        self.desk_corner.setCurrentIndex(max(0, self.desk_corner.findData(self.result_settings.widget_corner)))
        desk_row.addWidget(self.desk_corner)
        v.addLayout(desk_row)
        from jarvis.ui import screens

        mons = screens.screens()
        self.desk_screen = QComboBox()
        for i, sc in enumerate(mons):
            self.desk_screen.addItem(screens.label(sc, i), "" if sc is screens.main_screen() else sc.name())
        self.desk_screen.setCurrentIndex(max(0, self.desk_screen.findData(self.result_settings.widget_screen)))
        self.popup_screen = QComboBox()
        for key, text in (("mouse", "Where my mouse is"), ("main", "Main screen"), ("jarvis", "Where Jarvis is")):
            self.popup_screen.addItem(text, key)
        self.popup_screen.setCurrentIndex(max(0, self.popup_screen.findData(self.result_settings.popup_screen)))
        if len(mons) > 1:  # the choice only matters with two or more monitors
            mon_row = QHBoxLayout()
            mon_row.addWidget(QLabel("Widget on"))
            mon_row.addWidget(self.desk_screen, 1)
            mon_row.addWidget(QLabel("Pop-ups on"))
            mon_row.addWidget(self.popup_screen, 1)
            v.addLayout(mon_row)
        v.addStretch()
        return w

    def _set_color(self, color: str) -> None:
        self.color = valid_color(color)
        self.color_label.setText(f"Selected: {self.color}")
        self.color_label.setStyleSheet(f"color:{self.color}; font-weight:700;")

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(QColor(self.color), self, "Accent colour")
        if c.isValid():
            self._set_color(c.name())

    def _audio_tab(self) -> QWidget:
        from jarvis.audio import list_devices

        s = self.result_settings
        w = QWidget()
        form = QFormLayout(w)
        self.mic = QComboBox()
        self.spk = QComboBox()
        self.mic.addItem("System default", "")
        self.spk.addItem("System default", "")
        try:
            devices = list_devices()
        except Exception:
            devices = []
        # one host API keeps the list short (each device shows up once per Windows audio API)
        preferred = "MME" if platform.system() == "Windows" else None
        seen_in, seen_out = set(), set()
        for d in devices:
            if preferred and d["host"] != preferred:
                continue
            name = d["name"].strip()
            if "mapper" in name.lower() or "primary sound" in name.lower():
                continue
            if d["inputs"] and name not in seen_in:
                seen_in.add(name)
                self.mic.addItem(name, name)
            if d["outputs"] and name not in seen_out:
                seen_out.add(name)
                self.spk.addItem(name, name)
        for combo, current in ((self.mic, s.input_device), (self.spk, s.output_device)):
            idx = combo.findData(current)
            if idx < 0 and current:
                combo.addItem(current, current)
                idx = combo.count() - 1
            combo.setCurrentIndex(max(idx, 0))
        form.addRow("Microphone", self.mic)
        form.addRow("Speakers", self.spk)
        self.interrupt = QCheckBox("Let me talk over Jarvis (use headphones, or it hears itself)")
        self.interrupt.setChecked(s.allow_interrupt)
        form.addRow("", self.interrupt)

        form.addRow(QLabel("Wake word", objectName="section"))
        self.wake = QComboBox()
        for ww in WAKE_WORDS:
            self.wake.addItem("Off - always listening" if not ww else ww.replace("_", " ").title(), ww)
        idx = self.wake.findData(s.wake_word)
        self.wake.setCurrentIndex(max(idx, 0))
        form.addRow("Wake word", self.wake)
        self.name_wake = QCheckBox("Wake up when I say the assistant's name (any name, any language)")
        self.name_wake.setChecked(s.name_wake)
        form.addRow("", self.name_wake)
        self.sleep_after = QSpinBox()
        self.sleep_after.setRange(20, 3600)
        self.sleep_after.setSuffix(" s")
        self.sleep_after.setValue(s.sleep_after_seconds)
        form.addRow("Sleep after silence", self.sleep_after)
        note = QLabel("The offline wake word runs on your computer: while asleep no audio leaves it. "
                      "\"Say the name\" works without a wake word model but keeps the online session "
                      "listening while asleep. Say \"so jao\" / \"go to sleep\" to put Jarvis (not the PC) to sleep.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        form.addRow("", note)

        form.addRow(QLabel("Sound effects", objectName="section"))
        self.sfx = QCheckBox("Play sounds on wake, sleep, actions and reminders")
        self.sfx.setChecked(s.sound_effects)
        form.addRow("", self.sfx)
        self.sfx_volume = QSpinBox()
        self.sfx_volume.setRange(0, 100)
        self.sfx_volume.setSuffix(" %")
        self.sfx_volume.setValue(s.sound_volume)
        form.addRow("Effects volume", self.sfx_volume)
        return w

    def _features_tab(self) -> QWidget:
        from jarvis.tools.monitors import autostart_enabled

        s = self.result_settings
        w = QWidget()
        form = QFormLayout(w)
        self.briefing = QCheckBox("Morning briefing on the first start of the day")
        self.briefing.setChecked(s.morning_briefing)
        form.addRow(self.briefing)
        self.proactive = QCheckBox("Friendly check-in after a quiet spell")
        self.proactive.setChecked(s.proactive)
        form.addRow(self.proactive)
        self.proactive_minutes = QSpinBox()
        self.proactive_minutes.setRange(10, 600)
        self.proactive_minutes.setSuffix(" min")
        self.proactive_minutes.setValue(s.proactive_minutes)
        form.addRow("Check-in after", self.proactive_minutes)
        self.hw = QCheckBox("Warn about high CPU / RAM and low battery")
        self.hw.setChecked(s.hardware_alerts)
        form.addRow(self.hw)
        self.recap = QCheckBox("Evening recap of my day")
        self.recap.setChecked(s.evening_recap)
        form.addRow(self.recap)
        self.recap_hour = QSpinBox()
        self.recap_hour.setRange(17, 23)
        self.recap_hour.setSuffix(":00")
        self.recap_hour.setValue(s.recap_hour)
        form.addRow("Recap from", self.recap_hour)
        self.health = QCheckBox("Health nudges: eye breaks and water while I use the PC")
        self.health.setChecked(s.health_nudges)
        form.addRow(self.health)
        self.eye_minutes = QSpinBox()
        self.eye_minutes.setRange(20, 180)
        self.eye_minutes.setSuffix(" min")
        self.eye_minutes.setValue(s.eye_break_minutes)
        form.addRow("Eye break every", self.eye_minutes)
        self.water_minutes = QSpinBox()
        self.water_minutes.setRange(30, 240)
        self.water_minutes.setSuffix(" min")
        self.water_minutes.setValue(s.water_minutes)
        form.addRow("Water every", self.water_minutes)
        self.offline_voice = QCheckBox("Without internet: answer with the Windows voice and understand simple "
                                       "English voice commands (\"Jarvis, set a timer for 5 minutes\")")
        self.offline_voice.setChecked(s.offline_voice)
        form.addRow(self.offline_voice)
        self._calendar_rows(form)
        self.autostart = QCheckBox("Start Jarvis when Windows starts")
        self.autostart.setEnabled(platform.system() == "Windows")
        self.autostart.setChecked(autostart_enabled())
        form.addRow(self.autostart)
        topics = self.app.state.get("watch_topics", [])
        t = QLabel("Watched news topics: " + (", ".join(topics) if topics else "none (say \"watch news about ...\")"))
        t.setObjectName("muted")
        t.setWordWrap(True)
        form.addRow(t)
        return w

    def _calendar_rows(self, form: QFormLayout) -> None:
        s = self.result_settings
        form.addRow(QLabel("CALENDAR", objectName="section"))
        self.cal_links = QLineEdit(" ".join(s.calendar_urls))
        self.cal_links.setEchoMode(QLineEdit.EchoMode.Password)
        self.cal_links.setPlaceholderText("Private iCal link(s): https://…/basic.ics or webcal://…")
        self.cal_links.setToolTip("Google Calendar: Settings > your calendar > Integrate calendar > Secret address in "
                                  "iCal format.\nOutlook: Settings > Calendar > Shared calendars > Publish > ICS.\n"
                                  "Several links: separate them with spaces. They're encrypted on this PC.")
        show = QPushButton("Show")
        show.setCheckable(True)
        show.toggled.connect(lambda on: self.cal_links.setEchoMode(
            QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        row = QHBoxLayout()
        row.addWidget(self.cal_links)
        row.addWidget(show)
        form.addRow("iCal links", row)
        self.cal_minutes = QSpinBox()
        self.cal_minutes.setRange(0, 60)
        self.cal_minutes.setSpecialValueText("off")
        self.cal_minutes.setSuffix(" min before")
        self.cal_minutes.setValue(s.calendar_reminder_minutes)
        form.addRow("Meeting reminder", self.cal_minutes)
        self.cal_status = QLabel(self._calendar_status(), objectName="muted")
        self.cal_status.setWordWrap(True)
        form.addRow(self.cal_status)

    def _calendar_status(self) -> str:
        from jarvis.tools import calendar_sync as cal

        if not cal.links(self.result_settings):
            return "Read-only: Jarvis reads your meetings and reminds you before they start. It can't change them."
        data = cal.data_file().load()
        if data.get("links") != cal.links_key(cal.links(self.result_settings)):
            return "Syncing in a moment…"
        lines = []
        for v in data.get("sources", {}).values():
            if v.get("ok"):
                lines.append(f"✓ {v.get('name')} · {v.get('count', 0)} events")
            else:
                lines.append(f"⚠ {v.get('name')}: {v.get('error')}")
        return "\n".join(lines) or "Syncing in a moment…"

    def _calendar_links(self) -> list[str] | None:
        """The typed links, or None (with the reason shown) when one can't be used."""
        from types import SimpleNamespace

        from jarvis.tools import calendar_sync as cal

        urls = cal.links(SimpleNamespace(calendar_urls=[self.cal_links.text()]))
        for u in urls:
            problem = cal.check_link(u)
            if problem:
                self.cal_status.setText("⚠ " + problem)
                return None
        return urls

    def _memory_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel("Everything Jarvis remembers about you. Stored only on this computer.", objectName="muted"))
        self.mem_table = QTableWidget(0, 4)
        self.mem_table.setHorizontalHeaderLabels(["Fact", "Value", "Learned", ""])
        self.mem_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.mem_table.verticalHeader().setVisible(False)
        self.mem_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        v.addWidget(self.mem_table)
        self._fill_memory()
        return w

    def _fill_memory(self) -> None:
        facts = self.app.memory.all()
        self.mem_table.setRowCount(len(facts))
        for row, f in enumerate(facts):
            self.mem_table.setItem(row, 0, QTableWidgetItem(f.key))
            self.mem_table.setItem(row, 1, QTableWidgetItem(f.value))
            when = datetime.fromtimestamp(f.updated).strftime("%d %b %Y") if f.updated else ""
            self.mem_table.setItem(row, 2, QTableWidgetItem(when))
            forget = QPushButton("Forget")
            forget.clicked.connect(lambda _, k=f.key: self._forget(k))
            self.mem_table.setCellWidget(row, 3, forget)

    def _forget(self, key: str) -> None:
        self.app.memory.forget(key)
        self._fill_memory()

    def _tools_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        tools = list(self.app.registry.tools.values())
        v.addWidget(QLabel(f"{len(tools)} tools. Add your own by dropping a .py file into the plugins folder.",
                           objectName="muted"))
        table = QTableWidget(len(tools), 2)
        table.setHorizontalHeaderLabels(["Tool", "What it does"])
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, t in enumerate(tools):
            table.setItem(row, 0, QTableWidgetItem(t.name))
            item = QTableWidgetItem(t.description)
            item.setToolTip(t.description)
            table.setItem(row, 1, item)
        v.addWidget(table)
        for err in self.app.registry.errors:
            e = QLabel(f"⚠ {err}")
            e.setStyleSheet("color:#ffb547;")
            v.addWidget(e)
        return w

    # -- save -----------------------------------------------------------------
    def _save(self) -> None:
        s = self.result_settings
        s.api_key = self.api_key.text().strip()
        s.user_name = self.user_name.text().strip()
        s.assistant_name = self.assistant_name.text().strip() or "Jarvis"
        s.voice = self.voice.currentData() or "Charon"
        s.speaking_style = self.style.currentData() or "friendly"
        s.expressive_voice = self.expressive.isChecked()
        s.reply_language = self.reply_lang.currentData() or "auto"
        s.reminder_language = self.reminder_lang.currentData() or "same"
        s.city = self.city.text().strip()
        s.live_model = self.live_model.text().strip() or Settings().live_model
        s.text_model = self.text_model.text().strip() or Settings().text_model
        s.theme_color = self.color
        s.theme_name = self.theme_name.currentText()
        s.theme_mode = self.theme_mode.currentData() or "dark"
        s.input_device = self.mic.currentData() or ""
        s.output_device = self.spk.currentData() or ""
        s.allow_interrupt = self.interrupt.isChecked()
        s.wake_word = self.wake.currentData() or ""
        s.sleep_after_seconds = self.sleep_after.value()
        s.name_wake = self.name_wake.isChecked()
        s.sound_effects = self.sfx.isChecked()
        s.sound_volume = self.sfx_volume.value()
        s.orb_face = self.orb_face.isChecked()
        s.power_up_wake = self.power_up.isChecked()
        s.desktop_widget = self.desk_widget.isChecked()
        s.widget_corner = self.desk_corner.currentData() or "top_right"
        s.widget_screen = self.desk_screen.currentData() or ""
        s.popup_screen = self.popup_screen.currentData() or "mouse"
        s.morning_briefing = self.briefing.isChecked()
        s.proactive = self.proactive.isChecked()
        s.proactive_minutes = self.proactive_minutes.value()
        s.hardware_alerts = self.hw.isChecked()
        s.evening_recap = self.recap.isChecked()
        s.recap_hour = self.recap_hour.value()
        s.health_nudges = self.health.isChecked()
        s.eye_break_minutes = self.eye_minutes.value()
        s.water_minutes = self.water_minutes.value()
        s.offline_voice = self.offline_voice.isChecked()
        urls = self._calendar_links()
        if urls is None:
            self.tabs.setCurrentIndex(3)
            self.cal_links.setFocus()
            return
        s.calendar_urls = urls
        s.calendar_reminder_minutes = self.cal_minutes.value()
        if not s.effective_api_key:
            self.tabs.setCurrentIndex(0)
            self.api_key.setFocus()
            self.api_key.setPlaceholderText("An API key is required")
            return
        if self.autostart.isEnabled():
            from jarvis.tools.monitors import autostart_enabled, set_autostart

            if self.autostart.isChecked() != autostart_enabled():
                set_autostart(self.autostart.isChecked())
        self.privacy.apply_to(s)
        self.accept()


def apply_settings(target: Settings, new: Settings) -> set[str]:
    """Copy new values into the shared Settings object; return the names that changed."""
    changed = set()
    for name, value in vars(new).items():
        if getattr(target, name) != value:
            setattr(target, name, value)
            changed.add(name)
    return changed

