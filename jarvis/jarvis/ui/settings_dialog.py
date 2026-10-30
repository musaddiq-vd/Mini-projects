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
from jarvis.config import VOICES, Settings
from jarvis.ui.theme import valid_color

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
        self.voice.addItems(VOICES)
        self.voice.setCurrentText(s.voice)
        form.addRow("Voice", self.voice)
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
        v.addWidget(QLabel("Accent colour", objectName="section"))
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
        self.sleep_after = QSpinBox()
        self.sleep_after.setRange(20, 3600)
        self.sleep_after.setSuffix(" s")
        self.sleep_after.setValue(s.sleep_after_seconds)
        form.addRow("Sleep after silence", self.sleep_after)
        note = QLabel("The wake word runs fully offline. While asleep no audio leaves your computer.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        form.addRow("", note)
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
        s.voice = self.voice.currentText()
        s.city = self.city.text().strip()
        s.live_model = self.live_model.text().strip() or Settings().live_model
        s.text_model = self.text_model.text().strip() or Settings().text_model
        s.theme_color = self.color
        s.input_device = self.mic.currentData() or ""
        s.output_device = self.spk.currentData() or ""
        s.allow_interrupt = self.interrupt.isChecked()
        s.wake_word = self.wake.currentData() or ""
        s.sleep_after_seconds = self.sleep_after.value()
        s.morning_briefing = self.briefing.isChecked()
        s.proactive = self.proactive.isChecked()
        s.proactive_minutes = self.proactive_minutes.value()
        s.hardware_alerts = self.hw.isChecked()
        if not s.effective_api_key:
            self.tabs.setCurrentIndex(0)
            self.api_key.setFocus()
            self.api_key.setPlaceholderText("An API key is required")
            return
        if self.autostart.isEnabled():
            from jarvis.tools.monitors import autostart_enabled, set_autostart

            if self.autostart.isChecked() != autostart_enabled():
                set_autostart(self.autostart.isChecked())
        self.accept()


def apply_settings(target: Settings, new: Settings) -> set[str]:
    """Copy new values into the shared Settings object; return the names that changed."""
    changed = set()
    for name, value in vars(new).items():
        if getattr(target, name) != value:
            setattr(target, name, value)
            changed.add(name)
    return changed

