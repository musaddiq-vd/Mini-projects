"""Builds the main window's widgets: the orb panel on the left and the chat panel on the right."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QStackedWidget, QVBoxLayout

from jarvis import __version__
from jarvis.ui.chat_view import ChatView
from jarvis.ui.fx import AnimatedBackground, ToastHost, add_glow
from jarvis.ui.hud import StatsBar, StatusDot, TypingDots
from jarvis.ui.mini import OrbGroup
from jarvis.ui.orb import Orb
from jarvis.ui.theme import WARN

QUICK_ACTIONS = [
    ("🖥  Screen", "What's on my screen?"),
    ("🌦  Weather", "What's the weather today?"),
    ("📰  News", "Give me today's top news in two lines."),
    ("📝  My list", "What's on my to-do list, and do I have any reminders or timers?"),
    ("⏱  5 min", "Set a 5 minute timer."),
    ("😄  Joke", "Tell me a short joke."),
]


class LayoutMixin:
    """Creates every widget of the main window."""

    # -- layout -------------------------------------------------------------------
    def _build(self) -> None:
        root = AnimatedBackground(self.accent)
        root.setObjectName("root")
        self.background = root
        outer = QHBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        # left: orb + captions + HUD + controls
        left = QFrame(objectName="panel")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(18, 18, 18, 14)
        self.title = QLabel(objectName="title")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lv.addWidget(self.title)
        self.clock = QLabel(objectName="muted")
        self.clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clock.setWordWrap(True)
        lv.addWidget(self.clock)
        pill_row = QHBoxLayout()
        pill_row.addStretch(1)
        self.net_pill = QLabel("")
        self.net_pill.setToolTip("Connection to the AI")
        pill_row.addWidget(self.net_pill)
        from jarvis.ui.meeting_pill import MeetingPill

        self.meeting_pill = MeetingPill()  # red REC pill while a meeting is recorded
        pill_row.addWidget(self.meeting_pill)
        pill_row.addStretch(1)
        lv.addLayout(pill_row)
        self._set_net("offline" if not self.app.settings.effective_api_key else "connecting")
        self.big_orb = Orb(self.accent, face=self.app.settings.orb_face)
        self.orb = OrbGroup(self.big_orb)  # every orb call also reaches the mini orb
        lv.addWidget(self.big_orb, 1)

        state_row = QHBoxLayout()
        state_row.addStretch(1)
        self.dot = StatusDot()
        state_row.addWidget(self.dot)
        self.state_label = QLabel("CONNECTING", objectName="state")
        state_row.addWidget(self.state_label)
        state_row.addStretch(1)
        lv.addLayout(state_row)
        self.hint = QLabel("", objectName="muted")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lv.addWidget(self.hint)

        self.caption = QLabel("", objectName="caption")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setWordWrap(True)
        self.caption.setFixedHeight(44)
        lv.addWidget(self.caption)

        self.stats = StatsBar(self.accent)
        lv.addWidget(self.stats)
        from jarvis.ui.today import TodayStrip

        self.today = TodayStrip(self.app, lambda: self.big_orb.timers, lambda q: self._send(q))
        self.today.on_badges = lambda: self.toggle_badges()
        lv.addWidget(self.today)

        from jarvis.ui.flow import flow_widget

        controls_box, controls = flow_widget(6, center=True)  # 2 rows when the window is narrow
        self.mute_btn = QPushButton("🎙  Mute")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setToolTip("Mute the microphone (Ctrl+M)")
        self.mute_btn.toggled.connect(self._toggle_mute)
        self.sleep_btn = QPushButton("💤  Sleep")
        self.sleep_btn.setToolTip("Put Jarvis to sleep / wake it up (Ctrl+Space)")
        self.sleep_btn.clicked.connect(self._toggle_sleep)
        settings_btn = QPushButton("⚙  Settings")
        settings_btn.setToolTip("Settings (Ctrl+,)")
        settings_btn.clicked.connect(self.open_settings)
        mini_btn = QPushButton("🗗  Mini")
        mini_btn.setToolTip("Floating mini orb that stays on top (Ctrl+Shift+M)")
        mini_btn.clicked.connect(self.toggle_mini)
        for b in (self.mute_btn, self.sleep_btn, mini_btn, settings_btn):
            controls.addWidget(b)
        lv.addWidget(controls_box)
        self.practice_btn = QPushButton("🎓  English Practice", objectName="primary")
        self.practice_btn.setToolTip("Practice speaking, listening, reading and grammar with your tutor. "
                                     "Jarvis's computer commands are off until you come back.")
        self.practice_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.practice_btn.clicked.connect(lambda: self.set_practice(not self.practicing))
        lv.addWidget(self.practice_btn)
        credit = QLabel(f"Jarvis v{__version__} · by Musaddiq Khan", objectName="muted")
        credit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        credit.setWordWrap(True)
        lv.addWidget(credit)
        outer.addWidget(left, 5)

        # right: header + banner + conversation + quick actions + input
        right = QFrame(objectName="panel")
        self.right_panel = right
        rv = QVBoxLayout(right)
        rv.setContentsMargins(18, 18, 18, 18)
        header = QHBoxLayout()
        self.header_label = QLabel("CONVERSATION", objectName="section")
        header.addWidget(self.header_label)
        header.addStretch(1)
        dash_btn = QPushButton("📊", objectName="small")
        dash_btn.setToolTip("Dashboard: week, habits, money, goals (Ctrl+D)")
        dash_btn.clicked.connect(lambda: self.toggle_dashboard())
        header.addWidget(dash_btn)
        planner_btn = QPushButton("📋", objectName="small")
        planner_btn.setToolTip("Planner: to-dos, reminders, timers, routines (Ctrl+P)")
        planner_btn.clicked.connect(self.toggle_planner)
        header.addWidget(planner_btn)
        badges_btn = QPushButton("🏆", objectName="small")
        badges_btn.setToolTip("Achievements")
        badges_btn.clicked.connect(self.toggle_badges)
        header.addWidget(badges_btn)
        help_btn = QPushButton("?", objectName="small")
        help_btn.setToolTip("What can I say? (F1)")
        help_btn.clicked.connect(self.toggle_help)
        header.addWidget(help_btn)
        search_btn = QPushButton("🔍", objectName="small")
        search_btn.setToolTip("Search the chat (Ctrl+F)")
        search_btn.clicked.connect(self.open_search)
        header.addWidget(search_btn)
        export_btn = QPushButton("⤓", objectName="small")
        export_btn.setToolTip("Save the conversation to Documents > Jarvis Chats")
        export_btn.clicked.connect(self.export_chat)
        header.addWidget(export_btn)
        self.mode_btn = QPushButton("🌙", objectName="small")
        self.mode_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mode_btn.clicked.connect(self.cycle_theme_mode)
        header.addWidget(self.mode_btn)
        clear = QPushButton("🧹 Clear", objectName="small")
        clear.setToolTip("Clear the conversation view (Ctrl+L)")
        clear.clicked.connect(self._clear_chat)
        header.addWidget(clear)
        rv.addLayout(header)

        self.banner = QFrame(objectName="banner")
        bl = QHBoxLayout(self.banner)
        self.banner_text = QLabel()
        self.banner_text.setWordWrap(True)
        self.banner_text.setStyleSheet(f"color:{WARN}; background:transparent; font-weight:600;")
        bl.addWidget(self.banner_text, 1)
        self.banner_yes = QPushButton("Confirm", objectName="danger")
        self.banner_no = QPushButton("Cancel")
        bl.addWidget(self.banner_yes)
        bl.addWidget(self.banner_no)
        self.banner_yes.clicked.connect(lambda: self._answer_confirm(True))
        self.banner_no.clicked.connect(lambda: self._answer_confirm(False))
        self.banner.hide()
        self._pending_reply = None
        self._banner_timer = QTimer(self)
        self._banner_timer.setSingleShot(True)
        self._banner_timer.timeout.connect(lambda: self._answer_confirm(False, timed_out=True))
        rv.addWidget(self.banner)

        self.search_bar = QFrame(objectName="card")
        sb = QHBoxLayout(self.search_bar)
        sb.setContentsMargins(10, 4, 6, 4)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search messages...")
        self.search.textChanged.connect(self._on_search)
        self.search.setStyleSheet("background: transparent; border: none; padding: 4px;")
        sb.addWidget(QLabel("🔍"))
        sb.addWidget(self.search, 1)
        self.search_count = QLabel("", objectName="muted")
        sb.addWidget(self.search_count)
        close_search = QPushButton("✕", objectName="small")
        close_search.clicked.connect(self.close_search)
        sb.addWidget(close_search)
        self.search_bar.hide()
        rv.addWidget(self.search_bar)
        self.chat = ChatView()
        self.practice = None  # the practice screen is built the first time practice mode is turned on
        self.stack = QStackedWidget()
        self.stack.addWidget(self.chat)
        rv.addWidget(self.stack, 1)
        self.typing = TypingDots()
        self.typing.hide()
        rv.addWidget(self.typing)

        from jarvis.ui.flow import flow_widget

        self.chips, chips = flow_widget(6)  # wraps to a second line in a narrow window
        for label, prompt in QUICK_ACTIONS:
            b = QPushButton(label, objectName="chip")
            b.setToolTip(prompt)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _, t=prompt: self._send(t))
            chips.addWidget(b)
        rv.addWidget(self.chips)

        self.doc_chip = QPushButton(objectName="chip")
        self.doc_chip.setToolTip("Questions you ask now are answered from this document. Click to close it.")
        self.doc_chip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.doc_chip.clicked.connect(lambda: self.app.registry.call("close_document", {}, self.app.ctx))
        self.doc_chip.hide()
        rv.addWidget(self.doc_chip, 0, Qt.AlignmentFlag.AlignLeft)

        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type a message, or just talk...  (Ctrl+K)")
        self.input.returnPressed.connect(self._send_typed)
        self.input.installEventFilter(self)
        self._history: list[str] = []
        self._history_pos = 0
        send = QPushButton("Send ➤", objectName="primary")
        send.clicked.connect(self._send_typed)
        row.addWidget(self.input, 1)
        row.addWidget(send)
        rv.addLayout(row)
        outer.addWidget(right, 6)
        self.setCentralWidget(root)
        self.toasts = ToastHost(root)
        from jarvis.lang import reminder_language
        from jarvis.ui.reminder_popup import ReminderPopups

        self.reminders = ReminderPopups(
            lambda: self.accent,
            on_done=lambda rid: self._reminder_action("reminder_done", {"id": rid}),
            on_snooze=lambda rid, m: self._reminder_action("snooze_reminder", {"id": rid, "minutes": m}),
            on_cancel=lambda rid: self._reminder_action("cancel_reminder", {"id": rid}),
            on_open=self._show_window,
            language=lambda: reminder_language(self.app.settings, self.app.state),
            screen=lambda: __import__("jarvis.ui.screens", fromlist=["x"]).for_popups(
                self.app.settings.popup_screen, self))
        self._glows = [add_glow(b, self.accent, 20 if b.objectName() != "chip" else 14)
                       for b in root.findChildren(QPushButton)]
        self._glows.append(add_glow(self.input, self.accent, 18))

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._theme_timer = QTimer(self)
        self._theme_timer.timeout.connect(self._check_auto_theme)
        self._theme_timer.start(60_000)
        self._clock_timer.start(1000)
        self._update_clock()
