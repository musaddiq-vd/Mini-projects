"""Chat glue of the main window: sending, history, search, export, side panels and confirmations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt

from jarvis.ui import theme
from jarvis.ui.theme import OK

DOC_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".xlsm", ".txt", ".md", ".csv"}


class ChatMixin:
    """Everything between the input line, the chat view and the side panels."""

    # -- input history and drag & drop ---------------------------------------------
    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        from PyQt6.QtCore import QEvent

        if obj is self.input and event.type() == QEvent.Type.KeyPress and self._history:
            if event.key() == Qt.Key.Key_Up:
                self._history_pos = max(0, self._history_pos - 1)
            elif event.key() == Qt.Key.Key_Down:
                self._history_pos = min(len(self._history), self._history_pos + 1)
            else:
                return False
            hist = self._history + [""]
            self.input.setText(hist[self._history_pos])
            return True
        return False

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if not paths:
            return
        doc = next((p for p in paths if Path(p).suffix.lower() in DOC_SUFFIXES), None)
        if doc and len(paths) == 1:  # one document: open it for questions
            result = self.app.registry.call("open_document", {"path": doc}, self.app.ctx)
            if "error" in result:
                self._add_system(f"⚠ {result['error']}")
                return
            self._send(f"I opened the document '{Path(doc).name}'. Give me a two-sentence summary using "
                       "ask_document, then ask what I want to know about it.")
            return
        for path in paths[:3]:
            if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
                self.chat.add_image(path, Path(path).name, self.accent)
        names = "; ".join(paths[:3])
        self._send(f"I dropped this file into the window: {names}. Read or look at it and tell me "
                   f"briefly what it is, then ask what I want to do with it.")

    _welcome_row = None

    def _add_bubble(self, who: str, text: str, mine: bool = False, typewriter: bool = False) -> None:
        if not text or text.startswith("(system"):
            return  # app-generated instructions are not shown as user messages
        if self._welcome_row is not None:  # the conversation has started: the welcome card has done its job
            try:
                self._welcome_row.hide()
                self._welcome_row.deleteLater()
            except RuntimeError:
                pass
            self._welcome_row = None
        self.chat.add_message(who, text, mine, typewriter=typewriter)

    def toast(self, icon: str, text: str, color: str, seconds: float = 4.5) -> None:
        if self.isVisible() and not self.isMinimized():
            self.toasts.show(icon, text, color, seconds)

    def _add_system(self, text: str) -> None:
        self.chat.add_system(text)

    # -- help --------------------------------------------------------------------------
    def toggle_help(self) -> None:
        if getattr(self, "help", None) is None:
            from jarvis.ui.help import HelpPanel

            self.help = HelpPanel(self.right_panel, self._send)
        self.help.toggle()

    def toggle_dashboard(self, tab: str | None = None) -> None:
        if getattr(self, "dashboard", None) is None:
            from jarvis.ui.dashboard import DashboardPanel

            self.dashboard = DashboardPanel(self.right_panel, self.app, self._send)
        self.dashboard.toggle(tab)

    def toggle_planner(self) -> None:
        if getattr(self, "planner", None) is None:
            from jarvis.ui.planner import PlannerPanel

            self.planner = PlannerPanel(self.right_panel, self.app, lambda: self.big_orb.timers, self._send)
        self.planner.toggle()

    def toggle_badges(self) -> None:
        if getattr(self, "badges", None) is None:
            from jarvis.ui.badges import BadgesPanel

            self.badges = BadgesPanel(self.right_panel, self.app.achievements)
        self.badges.toggle()

    def _escape(self) -> None:
        if getattr(self, "dashboard", None) is not None and self.dashboard.isVisible():
            self.dashboard.hide()
        elif getattr(self, "planner", None) is not None and self.planner.isVisible():
            self.planner.hide()
        elif getattr(self, "badges", None) is not None and self.badges.isVisible():
            self.badges.hide()
        elif getattr(self, "help", None) is not None and self.help.isVisible():
            self.help.close_panel()
        else:
            self.close_search()

    # -- search and export -------------------------------------------------------
    def open_search(self) -> None:
        self.search_bar.show()
        self.search.setFocus()
        self.search.selectAll()

    def close_search(self) -> None:
        if not self.search_bar.isHidden():
            self.search.clear()
            self.search_bar.hide()
            self.chat.filter("")
            self.chat.scroll_to_end()

    def _on_search(self, text: str) -> None:
        n = self.chat.filter(text)
        self.search_count.setText(f"{n} match{'es' if n != 1 else ''}" if text.strip() else "")

    def export_chat(self) -> Path | None:
        items = self.history.recent(5000)
        if not items:
            self.toast("⤓", "Nothing to save yet.", theme.MUTED)
            return None
        folder = Path.home() / "Documents" / "Jarvis Chats"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"Jarvis chat {datetime.now():%Y-%m-%d %H-%M}.md"
        name = self.app.settings.assistant_name
        lines = [f"# Conversation with {name}", ""]
        last_day = None
        for item in items:
            when = datetime.fromtimestamp(item.get("t", 0) or 0)
            if when.date() != last_day:
                last_day = when.date()
                lines += ["", f"## {when:%A, %d %B %Y}", ""]
            who = "You" if item["role"] == "user" else name
            lines.append(f"**{who}** ({when:%I:%M %p}): {item['text']}  ")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.toast("⤓", f"Saved to Documents › Jarvis Chats ({len(items)} messages)", OK)
        return path

    def _clear_chat(self) -> None:
        self.chat.clear()
        self.history.clear()

    def _restore_history(self) -> None:
        items = self.history.recent(30)
        from jarvis.ui.welcome import WelcomeCard

        if not items:
            self.welcome = WelcomeCard(self.app, self._send)
            self._welcome_row = self.chat._add_row(self.welcome, "center", text="welcome", animate=False)
            return
        name = self.app.settings.assistant_name
        self.chat.animate = False
        for item in items:
            mine = item["role"] == "user"
            self.chat.add_message("You" if mine else name, item["text"], mine,
                                  when=datetime.fromtimestamp(item.get("t", 0) or 0))
        self.chat.add_system("──────  earlier conversation  ·  Ctrl+L clears it  ──────")
        self.welcome = WelcomeCard(self.app, self._send, compact=True)  # a fresh "welcome back" each start
        self._welcome_row = self.chat._add_row(self.welcome, "center", text="welcome", animate=False)
        self.chat.animate = True

    # -- controls ---------------------------------------------------------------
    def _send_typed(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        if not self._history or self._history[-1] != text:
            self._history = (self._history + [text])[-50:]
        self._history_pos = len(self._history)
        self._send(text)

    def _send(self, text: str) -> None:
        self.chat.scroll_to_end()
        if not self.assistant:
            if self._booting:  # still starting: keep it, it goes out in a moment
                self._typed_early = (self._typed_early + [text])[-10:]
                self._add_bubble("You", text, mine=True)
                return
            self._add_system("Not connected yet - check Settings.")
            return
        self.assistant.send_text(text)

    # -- confirmation -------------------------------------------------------------
    def _on_confirm(self, question: str, reply) -> None:
        self._pending_reply = reply
        self.banner_text.setText(f"⚠  {question}")
        self.banner.show()
        self._banner_timer.start(54_000)  # the tool stops waiting after ~55 s
        self.app.events.emit("confirm_needed", question=question)
        self._show_window()

    def _answer_confirm(self, yes: bool, timed_out: bool = False) -> None:
        self._banner_timer.stop()
        self.banner.hide()
        reply, self._pending_reply = self._pending_reply, None
        if reply:
            reply(yes)
        if timed_out:
            self._add_system("Confirmation timed out - cancelled.")
