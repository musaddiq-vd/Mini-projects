"""Privacy tab: see everything Jarvis keeps, delete any of it, and choose how much is kept."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from jarvis.config import home_dir

OUTSIDE = (
    "<b>What leaves your computer</b><br>"
    "• Your voice and messages go to <b>Google Gemini</b> so Jarvis can answer.<br>"
    "• The screen or camera only when you ask (you'll see a flash and a notice).<br>"
    "• Web and news searches go to <b>DuckDuckGo</b>; weather to <b>Open-Meteo</b> (city name only).<br>"
    "• If you link a calendar, Jarvis downloads it from your calendar provider (nothing is uploaded).<br>"
    "• Nothing else: no analytics, no tracking, no accounts.<br><br>"
    "<b>Protection</b><br>"
    "• Your API key and calendar links are encrypted with your Windows account.<br>"
    "• Guardrails never send or save passwords, OTPs, card, bank or ID numbers and keys, and mask them "
    "in history and logs.<br>"
    "• While asleep with the offline wake word, no audio leaves your PC."
)


def _size(path: Path) -> int:
    try:
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return path.stat().st_size if path.exists() else 0
    except OSError:
        return 0


def _count(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024 or unit == "MB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


class PrivacyTab(QWidget):
    def __init__(self, app, settings, parent=None):
        super().__init__(parent)
        self.app = app
        self.s = settings  # the dialog's working copy (saved with Save)
        self.rows: dict[str, tuple[QLabel, QPushButton]] = {}
        v = QVBoxLayout(self)

        v.addWidget(QLabel("WHAT JARVIS KEEPS ON THIS COMPUTER", objectName="section"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        items: list[tuple[str, str, Callable[[], None]]] = [
            ("history", "💬  Chat history", self._clear_history),
            ("memory", "🧠  Things Jarvis remembers about you", self._clear_memory),
            ("lists", "📝  Notes, reminders and routines", self._clear_lists),
            ("practice", "🎓  Language practice progress", self._clear_practice),
            ("money", "💰  Expenses and budgets", lambda: self._clear_file("expenses")),
            ("habits", "✅  Habits", lambda: self._clear_file("habits")),
            ("goals", "🎯  Goals", lambda: self._clear_file("goals")),
            ("calendar", "📅  Calendar copy (links stay in Settings)", lambda: self._clear_file("calendar")),
            ("cache", "🖼  Copies of what Jarvis looked at", self._clear_cache),
            ("logs", "📄  Logs (for fixing problems)", self._clear_logs),
        ]
        for row, (key, title, action) in enumerate(items):
            grid.addWidget(QLabel(title), row, 0)
            info = QLabel(objectName="muted")
            grid.addWidget(info, row, 1)
            btn = QPushButton("Delete")
            btn.clicked.connect(lambda _, k=key, t=title, a=action: self._confirm(t.split("  ", 1)[1], a))
            grid.addWidget(btn, row, 2)
            self.rows[key] = (info, btn)
        v.addLayout(grid)
        wipe = QPushButton("🗑  Delete everything (keeps your settings and API key)", objectName="danger")
        wipe.clicked.connect(lambda: self._confirm("everything Jarvis keeps about you", self._clear_all))
        v.addWidget(wipe)

        v.addSpacing(8)
        v.addWidget(QLabel("CHOICES", objectName="section"))
        form = QFormLayout()
        self.save_history = QCheckBox("Save the conversation on this computer")
        self.save_history.setChecked(self.s.save_history)
        form.addRow(self.save_history)
        self.history_days = QSpinBox()
        self.history_days.setRange(0, 3650)
        self.history_days.setSpecialValueText("forever")
        self.history_days.setSuffix(" days")
        self.history_days.setValue(self.s.history_days)
        form.addRow("Keep conversation for", self.history_days)
        self.ask_camera = QCheckBox("Ask me before Jarvis uses the camera")
        self.ask_camera.setChecked(self.s.ask_camera)
        form.addRow(self.ask_camera)
        self.notify_capture = QCheckBox("Flash and show a notice whenever Jarvis looks at my screen or camera")
        self.notify_capture.setChecked(self.s.notify_capture)
        form.addRow(self.notify_capture)
        v.addLayout(form)

        v.addSpacing(8)
        v.addWidget(QLabel("BACKUP", objectName="section"))
        brow = QHBoxLayout()
        make = QPushButton("💾  Make a backup…")
        make.setToolTip("Save everything Jarvis keeps (memory, chat, lists, money, habits, goals, practice) "
                        "in one file. Your API key and calendar links are not included.")
        make.clicked.connect(self._make_backup)
        brow.addWidget(make)
        restore_btn = QPushButton("♻  Restore from a backup…")
        restore_btn.clicked.connect(self._restore_backup)
        brow.addWidget(restore_btn)
        brow.addStretch(1)
        v.addLayout(brow)
        self.backup_info = QLabel(objectName="muted")
        self.backup_info.setWordWrap(True)
        v.addWidget(self.backup_info)

        box = QFrame(objectName="card")
        b = QVBoxLayout(box)
        text = QLabel(OUTSIDE)
        text.setWordWrap(True)
        b.addWidget(text)
        v.addWidget(box)
        v.addStretch(1)
        self.refresh()

    def apply_to(self, s) -> None:
        s.save_history = self.save_history.isChecked()
        s.history_days = self.history_days.value()
        s.ask_camera = self.ask_camera.isChecked()
        s.notify_capture = self.notify_capture.isChecked()

    # -- facts -----------------------------------------------------------------------------
    def refresh(self) -> None:
        home = home_dir()
        history = self.app.history
        n_msgs = len(history.recent(100_000)) if history else 0
        st = self.app.state
        lists = len(st.get("notes", [])) + len(st.get("reminders", [])) + len(st.get("routines", {}))
        cache = list((home / "cache").glob("seen_*.jpg"))
        facts = {
            "history": f"{n_msgs} message{'s' if n_msgs != 1 else ''} · {_human(_size(home / 'history.jsonl'))}",
            "memory": _count(len(self.app.memory.all()), "fact"),
            "lists": _count(lists, "item"),
            "practice": _human(_size(home / "english.json")) if (home / "english.json").exists() else "nothing yet",
            "cache": f"{_count(len(cache), 'image')} · {_human(sum(_size(f) for f in cache))}",
            "logs": _human(_size(home / "logs")),
            "money": self._file_fact("expenses", "items", "expense"),
            "habits": self._file_fact("habits", "habits", "habit"),
            "goals": self._file_fact("goals", "goals", "goal"),
            "calendar": self._file_fact("calendar", "events", "event"),
        }
        for key, (info, btn) in self.rows.items():
            info.setText(facts[key])
        from jarvis.backup import backups_dir

        autos = sorted(backups_dir().glob("auto-*.zip"))
        when = datetime.fromtimestamp(autos[-1].stat().st_mtime).strftime("%d %b %Y") if autos else "not yet"
        self.backup_info.setText(f"Jarvis also keeps a weekly automatic copy on this PC (last: {when}). "
                                 "Backup files contain personal data - keep them somewhere safe.")

    @staticmethod
    def _file_fact(name: str, key: str, word: str) -> str:
        from jarvis.datafile import store

        st = store(name, {key: []})
        if not st.exists():
            return "nothing yet"
        return f"{_count(len(st.load()[key]), word)} · {_human(_size(st.path))}"

    @staticmethod
    def _clear_file(name: str) -> None:
        from jarvis.datafile import store

        store(name, {}).delete()

    # -- backup -----------------------------------------------------------------------------
    def _make_backup(self) -> None:
        from jarvis.backup import default_name, make_backup

        start = Path.home() / "Documents" / default_name()
        path, _ = QFileDialog.getSaveFileName(self, "Save Jarvis backup", str(start), "Jarvis backup (*.zip)")
        if not path:
            return
        try:
            manifest = make_backup(Path(path))
        except OSError as exc:
            QMessageBox.warning(self, "Backup", f"Couldn't save the backup:\n{exc}")
            return
        from jarvis.backup import describe

        QMessageBox.information(self, "Backup saved", f"Saved to:\n{path}\n\n{describe(manifest)}")

    def _restore_backup(self) -> None:
        from jarvis.backup import BackupError, describe, read_backup, restore

        path, _ = QFileDialog.getOpenFileName(self, "Restore Jarvis backup", str(Path.home() / "Documents"),
                                              "Jarvis backup (*.zip)")
        if not path:
            return
        try:
            manifest, _files = read_backup(Path(path))
        except BackupError as exc:
            QMessageBox.warning(self, "Restore", str(exc))
            return
        answer = QMessageBox.question(
            self, "Restore this backup?",
            f"{describe(manifest)}\n\nThis replaces what Jarvis keeps now. A copy of your current data is saved "
            "first, and Jarvis restarts afterwards.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            restore(Path(path))
        except (BackupError, OSError) as exc:
            QMessageBox.warning(self, "Restore", f"Couldn't restore: {exc}")
            return
        self.app.reload_stores()  # nothing old can be written over the restored files before the restart
        QMessageBox.information(self, "Restored", "Your data is back. Jarvis will restart now.")
        self.app.events.emit("restart_requested")

    def _confirm(self, what: str, action: Callable[[], None]) -> None:
        answer = QMessageBox.question(self, "Delete?", f"Permanently delete {what}?\nThis can't be undone.",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        if answer == QMessageBox.StandardButton.Yes:
            action()
            self.refresh()

    # -- delete actions ------------------------------------------------------------------------
    def _clear_history(self) -> None:
        if self.app.history:
            self.app.history.clear()

    def _clear_memory(self) -> None:
        for fact in list(self.app.memory.all()):
            self.app.memory.forget(fact.key)

    def _clear_lists(self) -> None:
        try:
            from jarvis.tools import reminders

            for timer in list(reminders._timers.values()):
                timer.cancel()
            reminders._timers.clear()
        except Exception:
            pass
        for key, empty in (("notes", []), ("reminders", []), ("routines", {})):
            self.app.state.set(key, empty)

    def _clear_practice(self) -> None:
        path = home_dir() / "english.json"
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        self.app._english = None  # starts fresh next time practice is used

    def _clear_cache(self) -> None:
        from jarvis.tools.vision import clear_seen_cache

        clear_seen_cache()

    def _clear_logs(self) -> None:
        import logging

        for handler in logging.getLogger("jarvis").handlers:
            if hasattr(handler, "stream") and getattr(handler, "baseFilename", None):
                handler.stream.seek(0)
                handler.stream.truncate()
        for f in (home_dir() / "logs").glob("jarvis.log.*"):
            try:
                f.unlink()
            except OSError:
                pass
        log = home_dir() / "logs" / "jarvis.log"
        if log.exists() and not logging.getLogger("jarvis").handlers:
            log.write_text("", encoding="utf-8")

    def _clear_all(self) -> None:
        for fn in (self._clear_history, self._clear_memory, self._clear_lists, self._clear_practice,
                   self._clear_cache, self._clear_logs):
            fn()
        for name in ("expenses", "habits", "goals", "calendar"):
            self._clear_file(name)
        for key in ("last_session_summary", "seen_headlines", "watch_topics"):
            self.app.state.set(key, [] if key == "watch_topics" else ({} if key == "seen_headlines" else None))


