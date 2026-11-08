"""Settings > Phone: connect your own Telegram bot and pair your phone with a one-time code."""

from __future__ import annotations

import threading
from collections.abc import Callable

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

HOW = ("1. In Telegram, open <b>@BotFather</b>, send <b>/newbot</b> and pick a name.<br>"
       "2. Copy the token it gives you and paste it below, then press <b>Connect</b>.<br>"
       "3. Open your new bot on your phone, press <b>Start</b> and send the 6-digit code shown here.<br>"
       "Only your Telegram account will be able to use it. Risky actions ask you Yes / No in the chat.")


class PhoneTab(QWidget):
    def __init__(self, app, pair: Callable[[str], tuple[str, str]] | None = None, parent=None):
        """pair(token) -> (bot username, code); provided by the main window (it runs the bot)."""
        super().__init__(parent)
        self.app = app
        self.pair_fn = pair
        self._unpaired = False
        v = QVBoxLayout(self)
        v.addWidget(QLabel("PHONE (TELEGRAM)", objectName="section"))
        how = QLabel(HOW, objectName="muted")
        how.setTextFormat(Qt.TextFormat.RichText)
        how.setWordWrap(True)
        v.addWidget(how)
        row = QHBoxLayout()
        self.token = QLineEdit(app.settings.telegram_token)
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("Bot token from @BotFather (like 123456:ABC-...)")
        row.addWidget(self.token, 1)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_bot)
        row.addWidget(self.connect_btn)
        self.unpair_btn = QPushButton("Unpair")
        self.unpair_btn.clicked.connect(self.unpair)
        row.addWidget(self.unpair_btn)
        v.addLayout(row)
        self.code = QLabel("")
        self.code.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.code.setStyleSheet("font-size: 26pt; font-weight: 700; letter-spacing: 6px;")
        self.code.hide()
        v.addWidget(self.code)
        self.status = QLabel(objectName="muted")
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        v.addStretch(1)
        self._watch = QTimer(self)
        self._watch.timeout.connect(self._check_paired)
        self._show_state()

    def _show_state(self) -> None:
        s = self.app.settings
        if s.telegram_chat_id and not self._unpaired:
            self.status.setText(f"✅ Paired with {s.telegram_user or 'your phone'}. Send your bot a message any time.")
            self.unpair_btn.setEnabled(True)
        else:
            self.status.setText("Not connected yet." if not s.telegram_token else "Token saved - not paired yet.")
            self.unpair_btn.setEnabled(False)

    def connect_bot(self) -> None:
        token = self.token.text().strip()
        if ":" not in token or len(token) < 30:
            self.status.setText("⚠ That doesn't look like a bot token - copy it again from @BotFather.")
            return
        if self.pair_fn is None:
            self.status.setText("Save first, then connect from the running Jarvis window.")
            return
        self.connect_btn.setEnabled(False)
        self.status.setText("Checking the token…")
        result: dict = {}

        def work():
            try:
                result["bot"], result["code"] = self.pair_fn(token)
            except Exception as exc:
                result["error"] = str(exc)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()

        def wait():
            if thread.is_alive():
                QTimer.singleShot(150, wait)
                return
            self.connect_btn.setEnabled(True)
            if "error" in result:
                self.status.setText(f"⚠ {result['error']}")
                return
            self._unpaired = False
            self.code.setText(result["code"])
            self.code.show()
            self.status.setText(f"Now open <b>t.me/{result['bot']}</b> on your phone, press Start and send this "
                                "code (valid for 10 minutes).")
            self.status.setTextFormat(Qt.TextFormat.RichText)
            self._watch.start(1000)

        QTimer.singleShot(150, wait)

    def _check_paired(self) -> None:
        s = self.app.settings
        if s.telegram_chat_id:
            self._watch.stop()
            self.code.hide()
            self.status.setTextFormat(Qt.TextFormat.PlainText)
            self._show_state()

    def unpair(self) -> None:
        self._unpaired = True
        self.code.hide()
        self.status.setText("Will be unpaired when you press Save.")
        self.unpair_btn.setEnabled(False)

    def apply_to(self, s) -> None:
        live = self.app.settings  # pairing happens on the live settings while this dialog is open
        s.telegram_token = self.token.text().strip()
        if self._unpaired or s.telegram_token != live.telegram_token:
            s.telegram_chat_id, s.telegram_user = 0, ""
        else:
            s.telegram_chat_id, s.telegram_user = live.telegram_chat_id, live.telegram_user
